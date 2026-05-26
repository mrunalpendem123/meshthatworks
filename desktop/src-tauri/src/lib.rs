mod commands;
mod engine;
mod paths;

use engine::Engine;
use tauri::{Manager, RunEvent};

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_shell::init())
        .manage(Engine::default())
        .manage(commands::PairState::default())
        .setup(|app| {
            if cfg!(debug_assertions) {
                app.handle().plugin(
                    tauri_plugin_log::Builder::default()
                        .level(log::LevelFilter::Info)
                        .build(),
                )?;
            }

            // Single health poller for the whole app lifetime.
            engine::spawn_poller(app.handle().clone());

            // If SwiftLM + a model are already in place, bring the node up
            // automatically so the dashboard lands on a live engine.
            let handle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                if commands::setup_state(&handle).ready {
                    if let Some(eng) = handle.try_state::<Engine>() {
                        let _ = eng.start(&handle).await;
                    }
                }
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::get_setup_state,
            commands::list_installed_models,
            commands::set_active_model,
            commands::download_model,
            commands::start_engine,
            commands::stop_engine,
            commands::restart_engine,
            commands::set_split_mode,
            commands::split_status,
            commands::discovered_nodes,
            commands::node_status,
            commands::list_peers,
            commands::pair_start,
            commands::pair_cancel,
            commands::join_mesh,
            commands::run_setup,
            commands::chat_stream,
            commands::open_url,
        ])
        .build(tauri::generate_context!())
        .expect("error while building MeshThatWorks");

    app.run(|handle, event| {
        // Make sure the SwiftLM child dies with the app instead of lingering.
        if let RunEvent::ExitRequested { .. } = event {
            if let Some(eng) = handle.try_state::<Engine>() {
                let h = handle.clone();
                tauri::async_runtime::block_on(async move {
                    if let Some(eng) = h.try_state::<Engine>() {
                        let _ = eng.stop().await;
                    }
                });
                let _ = eng;
            }
        }
    });
}
