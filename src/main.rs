// Импортируем необходимые модули и типы
use std::time::Instant;
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::buffer::cpu_pool::CpuBufferPool;
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, SubpassContents};
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::device::{Device, DeviceExtensions, DeviceCreateInfo};
use vulkano::device::physical::PhysicalDevice;
use vulkano::device::QueueCreateInfo;
use vulkano::image::view::ImageView;
use vulkano::image::AttachmentImage;
use vulkano::format::Format;
use vulkano::instance::{Instance, InstanceExtensions, InstanceCreateInfo};
use vulkano::pipeline::{GraphicsPipeline, Pipeline, PipelineBindPoint};
use vulkano::pipeline::graphics::vertex_input::BuffersDefinition;
use vulkano::pipeline::graphics::input_assembly::{InputAssemblyState, PrimitiveTopology};
use vulkano::pipeline::graphics::rasterization::{RasterizationState, CullMode};
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::graphics::depth_stencil::DepthStencilState;
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, Subpass};
use vulkano::swapchain::{Swapchain, SwapchainCreateInfo};
use vulkano::sync::{self, GpuFuture};
use vulkano_win::VkSurfaceBuild;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;
use cgmath::{Matrix4, Point3, Vector3, Rad, perspective};
use bytemuck::{Pod, Zeroable};
use vulkano::swapchain::acquire_next_image;
use std::path::Path;
use tobj;
use tobj::LoadOptions;

// Шейдеры на GLSL, компилируемые в SPIR-V во время сборки
mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: "#version 450
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 0) out vec3 v_normal;
layout(set = 0, binding = 0) uniform Data { mat4 transform; mat4 model; } uniforms;
void main() {
    gl_Position = uniforms.transform * vec4(position, 1.0);
    v_normal = normalize((uniforms.model * vec4(normal, 0.0)).xyz);
}"
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: "#version 450
layout(location = 0) in vec3 v_normal;
layout(location = 0) out vec4 f_color;
void main() {
    vec3 light_dir = normalize(vec3(1.0, 1.0, 1.0));
    float diff = max(dot(normalize(v_normal), light_dir), 0.0);
    vec3 gray = vec3(0.5);
    float ambient = 0.2;
    vec3 color = gray * (ambient + diff * (1.0 - ambient));
    f_color = vec4(color, 1.0);
}"
    }
}

// Отмечаем uniform-данные как Pod и Zeroable для безопасного копирования в буфер
unsafe impl Zeroable for vs::ty::Data {}
unsafe impl Pod for vs::ty::Data {}

// Описание вершины: позиция и цвет
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
struct Vertex { position: [f32; 3], normal: [f32; 3] }
vulkano::impl_vertex!(Vertex, position, normal);

#[allow(unused_assignments)]
fn main() {
    // Цикл событий и инстанс Vulkan
    let event_loop = EventLoop::new();
    let instance = Instance::new(InstanceCreateInfo {
        enabled_extensions: InstanceExtensions::none().union(&vulkano_win::required_extensions()),
        ..Default::default()
    }).expect("Не удалось создать Vulkan-инстанс");
    let surface = WindowBuilder::new()
        .with_title("3D модель на Vulkan")
        .build_vk_surface(&event_loop, instance.clone())
        .unwrap();

    // Физическое устройство и очередь
    let physical = PhysicalDevice::enumerate(&instance)
        .next()
        .expect("Нет доступных устройств Vulkan");
    let queue_family = physical.queue_families()
        .find(|&q| q.supports_graphics() && q.supports_surface(&surface).unwrap_or(false))
        .expect("Нет подходящего семейства очередей");
    let (device, mut mut_queues) = Device::new(
        physical,
        DeviceCreateInfo {
            enabled_extensions: DeviceExtensions { khr_swapchain: true, ..DeviceExtensions::none() },
            queue_create_infos: vec![QueueCreateInfo::family(queue_family)],
            ..Default::default()
        }
    ).unwrap();
    let queue = mut_queues.next().unwrap();

    // Swapchain и изображения
    let (swapchain, images) = {
        let caps = physical.surface_capabilities(&surface, Default::default()).unwrap();
        let format = physical.surface_formats(&surface, Default::default()).unwrap()[0].0;
        let dimensions: [u32; 2] = surface.window().inner_size().into();
        Swapchain::new(
            device.clone(), surface.clone(), SwapchainCreateInfo {
                min_image_count: caps.min_image_count,
                image_format: Some(format),
                image_extent: dimensions,
                image_usage: vulkano::image::ImageUsage::color_attachment(),
                composite_alpha: caps.supported_composite_alpha.iter().next().unwrap(),
                ..Default::default()
            }
        ).unwrap()
    };

    // Рендер-проход и фреймбуферы
    let depth_buffer = AttachmentImage::transient(device.clone(), swapchain.image_extent(), Format::D16_UNORM).unwrap();
    let render_pass = vulkano::single_pass_renderpass!(
        device.clone(),
        attachments: {
            color: {
                load: Clear,
                store: Store,
                format: swapchain.image_format(),
                samples: 1,
            },
            depth: {
                load: Clear,
                store: DontCare,
                format: Format::D16_UNORM,
                samples: 1,
            }
        },
        pass: {
            color: [color],
            depth_stencil: {depth}
        }
    ).unwrap();
    let framebuffers = images.iter().map(|image| {
        let view = ImageView::new_default(image.clone()).unwrap();
        let depth_view = ImageView::new_default(depth_buffer.clone()).unwrap();
        Framebuffer::new(
            render_pass.clone(), FramebufferCreateInfo { attachments: vec![view, depth_view], ..Default::default() }
        ).unwrap()
    }).collect::<Vec<_>>();

    // Шейдеры
    let vs = vs::load(device.clone()).unwrap();
    let fs = fs::load(device.clone()).unwrap();

    // Графический конвейер
    let pipeline = GraphicsPipeline::start()
        .vertex_input_state(BuffersDefinition::new().vertex::<Vertex>())
        .vertex_shader(vs.entry_point("main").unwrap(), ())
        .input_assembly_state(InputAssemblyState::new())
        .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
        .fragment_shader(fs.entry_point("main").unwrap(), ())
        .rasterization_state(RasterizationState::new().cull_mode(CullMode::None))
        .depth_stencil_state(DepthStencilState::simple_depth_test())
        .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
        .build(device.clone()).unwrap();

    // Графический конвейер для линий (рёбер куба)
    let line_pipeline = GraphicsPipeline::start()
        .vertex_input_state(BuffersDefinition::new().vertex::<Vertex>())
        .vertex_shader(vs.entry_point("main").unwrap(), ())
        .input_assembly_state(InputAssemblyState::new().topology(PrimitiveTopology::LineList))
        .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
        .fragment_shader(fs.entry_point("main").unwrap(), ())
        .rasterization_state(RasterizationState::new())
        .depth_stencil_state(DepthStencilState::simple_depth_test())
        .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
        .build(device.clone()).unwrap();

    // Вершины и индексы куба
    let obj_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/models3d/monkey.obj");
    let load_options = LoadOptions { triangulate: true, single_index: true, ..Default::default() };
    let (models, _) = tobj::load_obj(obj_path.to_str().unwrap(), &load_options).expect("Не удалось загрузить OBJ");
    let mesh = &models[0].mesh;
    let vertices: Vec<Vertex> = (0..mesh.positions.len()/3).map(|i| {
        let pos = [mesh.positions[3*i], mesh.positions[3*i+1], mesh.positions[3*i+2]];
        let norm = if mesh.normals.len() >= mesh.positions.len() {
            [mesh.normals[3*i], mesh.normals[3*i+1], mesh.normals[3*i+2]]
        } else {
            [0.0, 0.0, 1.0]
        };
        Vertex { position: pos, normal: norm }
    }).collect();
    let indices: Vec<u32> = mesh.indices.iter().map(|&i| i as u32).collect();
    let vertex_buffer = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::vertex_buffer(), false, vertices.iter().cloned()).unwrap();
    let index_buffer = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::index_buffer(), false, indices.iter().cloned()).unwrap();

    // Пул униформ-буферов
    let uniform_buffer = CpuBufferPool::<vs::ty::Data>::uniform_buffer(device.clone());
    let rotation_start = Instant::now();
    let mut previous_frame_end = Some(sync::now(device.clone()).boxed());
    let mut recreate_swapchain = false;
    let mut swapchain = swapchain;
    let mut images = images;
    let mut framebuffers = framebuffers;
    let mut depth_buffer = depth_buffer;

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;
        match event {
            Event::WindowEvent { event: WindowEvent::Resized(_), .. } => {
                recreate_swapchain = true;
            }
            Event::RedrawEventsCleared => {
                if recreate_swapchain {
                    let dimensions: [u32; 2] = surface.window().inner_size().into();
                    let (new_swapchain, new_images) = match swapchain.recreate(SwapchainCreateInfo {
                        image_extent: dimensions,
                        ..swapchain.create_info()
                    }) {
                        Ok(r) => r,
                        Err(vulkano::swapchain::SwapchainCreationError::ImageExtentNotSupported { .. }) => return,
                        Err(e) => panic!("Ошибка пересоздания swapchain: {:?}", e)
                    };
                    swapchain = new_swapchain;
                    images = new_images;
                    depth_buffer = AttachmentImage::transient(device.clone(), swapchain.image_extent(), Format::D16_UNORM).unwrap();
                    framebuffers = images.iter().map(|image| {
                        let view = ImageView::new_default(image.clone()).unwrap();
                        let depth_view = ImageView::new_default(depth_buffer.clone()).unwrap();
                        Framebuffer::new(
                            render_pass.clone(), FramebufferCreateInfo { attachments: vec![view, depth_view], ..Default::default() }
                        ).unwrap()
                    }).collect::<Vec<_>>();
                    recreate_swapchain = false;
                }

                previous_frame_end.as_mut().unwrap().cleanup_finished();
                let elapsed = rotation_start.elapsed().as_secs_f32();
                let aspect = swapchain.image_extent()[0] as f32 / swapchain.image_extent()[1] as f32;
                let proj = perspective(Rad(std::f32::consts::FRAC_PI_2), aspect, 0.01, 100.0);
                let view = Matrix4::look_at_rh(Point3::new(2.0,2.0,2.0), Point3::new(0.0,0.0,0.0), Vector3::unit_z());
                let model = Matrix4::from_angle_y(Rad(elapsed));
                let mv = view * model;
                let uniform_data = vs::ty::Data { transform: (proj * mv).into(), model: mv.into() };
                let uniform_subbuffer = uniform_buffer.next(uniform_data).unwrap();
                let layout = pipeline.layout().set_layouts().get(0).unwrap();
                let set = PersistentDescriptorSet::new(
                    layout.clone(), [WriteDescriptorSet::buffer(0, uniform_subbuffer)]
                ).unwrap();
                let set_clone = set.clone();

                let (image_index, suboptimal, acquire_future) = match acquire_next_image(swapchain.clone(), None) {
                    Ok(r) => r,
                    Err(vulkano::swapchain::AcquireError::OutOfDate) => {
                        recreate_swapchain = true;
                        return;
                    }
                    Err(e) => panic!("Ошибка получения изображения: {:?}", e)
                };
                
                if suboptimal {
                    recreate_swapchain = true;
                }

                let mut builder = AutoCommandBufferBuilder::primary(
                    device.clone(), queue.family(), CommandBufferUsage::OneTimeSubmit
                ).unwrap();
                builder.begin_render_pass(
                    framebuffers[image_index].clone(), SubpassContents::Inline, vec![[0.0,0.0,0.1,1.0].into(), 1.0.into()]
                ).unwrap()
                    // Сначала рисуем треугольники
                    .bind_pipeline_graphics(pipeline.clone())
                    .set_viewport(0, [Viewport { origin: [0.0, 0.0], dimensions: [swapchain.image_extent()[0] as f32, swapchain.image_extent()[1] as f32], depth_range: 0.0..1.0 }])
                    .bind_vertex_buffers(0, vertex_buffer.clone())
                    .bind_index_buffer(index_buffer.clone())
                    .bind_descriptor_sets(PipelineBindPoint::Graphics, pipeline.layout().clone(), 0, set)
                    .draw_indexed(indices.len() as u32, 1, 0, 0, 0).unwrap()
                    .end_render_pass().unwrap();
                let command_buffer = builder.build().unwrap();

                let future = previous_frame_end.take().unwrap()
                    .join(acquire_future)
                    .then_execute(queue.clone(), command_buffer).unwrap()
                    .then_swapchain_present(queue.clone(), swapchain.clone(), image_index)
                    .then_signal_fence_and_flush();
                match future {
                    Ok(future) => {
                        previous_frame_end = Some(future.boxed());
                    }
                    Err(vulkano::sync::FlushError::OutOfDate) => {
                        recreate_swapchain = true;
                        previous_frame_end = Some(sync::now(device.clone()).boxed());
                    }
                    Err(e) => {
                        println!("Ошибка отображения: {:?}", e);
                        previous_frame_end = Some(sync::now(device.clone()).boxed());
                    }
                }
            }
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
                *control_flow = ControlFlow::Exit;
            }
            _ => ()
        }
    });
}
