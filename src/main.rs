mod model;
use eframe::egui;
use model::{ModelInfo, ForwardPassResult, LayerInfo};

// Simple Rust neural network implementation
#[derive(Clone)]
pub struct RustNeuralNetwork {
    pub layers: Vec<LayerInfo>,
    pub weights: Vec<Vec<Vec<f32>>>, // weights[layer][from][to]
}

impl RustNeuralNetwork {
    pub fn new(layers: &[LayerInfo], input_dim: usize) -> Self {
        let mut weights = Vec::new();
        let mut prev_dim = input_dim;
        for layer in layers.iter() {
            let out_dim = layer.output_dim.unwrap_or(1);
            // Random weights for demonstration
            let w = (0..prev_dim)
                .map(|_| (0..out_dim).map(|_| rand::random::<f32>() * 2.0 - 1.0).collect())
                .collect();
            weights.push(w);
            prev_dim = out_dim;
        }
        Self {
            layers: layers.to_vec(),
            weights,
        }
    }

    pub fn forward(&self, input: &[f32]) -> Vec<Vec<f32>> {
        let mut activations = vec![input.to_vec()];
        let mut x = input.to_vec();
        for (i, layer) in self.layers.iter().enumerate() {
            let w = &self.weights[i];
            let mut out = vec![0.0; layer.output_dim.unwrap_or(1)];
            for to in 0..out.len() {
                for from in 0..x.len() {
                    out[to] += x[from] * w[from][to];
                }
            }
            // No bias for simplicity
            // Activation
            match layer.activation.as_deref() {
                Some("relu") => {
                    for v in &mut out { *v = v.max(0.0); }
                }
                Some("sigmoid") => {
                    for v in &mut out { *v = 1.0 / (1.0 + (-*v).exp()); }
                }
                Some("tanh") => {
                    for v in &mut out { *v = v.tanh(); }
                }
                _ => {}
            }
            activations.push(out.clone());
            x = out;
        }
        activations
    }
}

struct MyApp {
    python_result: String,
    model_info: Option<ModelInfo>,
    selected_node: Option<(usize, usize)>,
    selected_edge: Option<(usize, usize, usize)>, // (layer, from_idx, to_idx)
    activations: Option<Vec<Vec<f32>>>,
    input_values: Vec<f32>,
    weights: Option<Vec<Vec<Vec<f32>>>>,

    // Editable model structure for user modifications
    editable_layers: Vec<LayerInfo>,
    editable_input_dim: usize,

    // Rust neural network
    rust_nn: Option<RustNeuralNetwork>,

    // Zoom and pan state
    zoom: f32,
    pan: egui::Vec2,
    is_panning: bool,
    last_pan_pos: Option<egui::Pos2>,
}

impl Default for MyApp {
    fn default() -> Self {
        let editable_layers = vec![
            LayerInfo {
                layer_type: "Dense".to_string(),
                input_dim: Some(2),
                output_dim: Some(3),
                activation: Some("relu".to_string()),
            },
            LayerInfo {
                layer_type: "Dense".to_string(),
                input_dim: Some(3),
                output_dim: Some(1),
                activation: Some("sigmoid".to_string()),
            },
        ];
        let editable_input_dim = 2;
        let rust_nn = Some(RustNeuralNetwork::new(&editable_layers, editable_input_dim));
        let model_info = Some(ModelInfo {
            layers: editable_layers.clone(),
        });
        Self {
            python_result: "Rust NN loaded".into(),
            model_info,
            selected_node: None,
            selected_edge: None,
            activations: None,
            input_values: vec![0.5, -0.2],
            weights: None,
            editable_layers,
            editable_input_dim,
            rust_nn,
            zoom: 1.0,
            pan: egui::Vec2::ZERO,
            is_panning: false,
            last_pan_pos: None,
        }
    }
}

impl MyApp {
    fn reload_rust_nn(&mut self) {
        // Update editable_layers' input_dim chain based on editable_input_dim
        if !self.editable_layers.is_empty() {
            self.editable_layers[0].input_dim = Some(self.editable_input_dim);
            for i in 1..self.editable_layers.len() {
                let prev_out = self.editable_layers[i - 1].output_dim;
                self.editable_layers[i].input_dim = prev_out;
            }
        }
        // Rebuild the Rust NN with updated layers
        self.rust_nn = Some(RustNeuralNetwork::new(&self.editable_layers, self.editable_input_dim));
        // Update model_info based on editable layers
        self.model_info = Some(ModelInfo {
            layers: self.editable_layers.clone(),
        });
        // Update status message to confirm application
        self.python_result = format!("Applied changes: input {}, {} layers", self.editable_input_dim, self.editable_layers.len());
        self.activations = None;
        self.weights = None;
        // Reset input values to match new input dimension
        self.input_values = vec![0.0; self.editable_input_dim];
        // Clear any previous selection
        self.selected_node = None;
        self.selected_edge = None;

        // --- Auto-zoom to fit the network in the bounding box ---
        // These values must match those in draw_network
        let margin = 40.0;
        let _node_radius = 12.0;
        let n_columns = self.editable_layers.len() + 1;
        let max_nodes = self.editable_layers.iter().map(|l| l.output_dim.unwrap_or(1)).max().unwrap_or(1).max(self.editable_input_dim);
        let fixed_layer_spacing = 120.0;
        let _layer_spacing = fixed_layer_spacing;
        let _edge_inset = 60.0;
        let total_graph_width = if n_columns > 1 {
            (n_columns - 1) as f32 * _layer_spacing
        } else {
            0.0
        };
        let total_graph_height = (max_nodes as f32 + 1.0) * 40.0; // 40.0 is the min node_spacing
        // Get the available rect (simulate, since we don't have ui here)
        // We'll use a default size, but you may want to pass the actual rect size from the UI
        let available_width = 800.0;
        let available_height = 400.0;
        let fit_zoom_x = (available_width - 2.0 * margin) / total_graph_width;
        let fit_zoom_y = (available_height - 2.0 * margin) / total_graph_height;
        let fit_zoom = fit_zoom_x.min(fit_zoom_y).min(1.0).max(0.2); // Don't zoom in past 1.0, or out past 0.2
        self.zoom = fit_zoom;
        // Center the graph
        let x_offset = (available_width - total_graph_width * self.zoom) / 2.0;
        let y_offset = (available_height - total_graph_height * self.zoom) / 2.0;
        self.pan = egui::vec2(x_offset - margin, y_offset - margin);
    }

    fn call_forward_pass(&mut self) -> Option<ForwardPassResult> {
        if let Some(nn) = &self.rust_nn {
            let acts = nn.forward(&self.input_values);
            Some(ForwardPassResult { activations: acts })
        } else {
            None
        }
    }

    fn call_get_weights(&mut self) -> Option<Vec<Vec<Vec<f32>>>> {
        self.rust_nn.as_ref().map(|nn| nn.weights.clone())
    }

    fn node_color(&self, layer_idx: usize, node_idx: usize) -> egui::Color32 {
        if self.selected_node == Some((layer_idx, node_idx)) {
            egui::Color32::YELLOW
        } else {
            egui::Color32::from_rgb(100, 200, 255)
        }
    }

    fn node_rect(&self, pos: egui::Pos2, node_radius: f32) -> egui::Rect {
        egui::Rect::from_center_size(pos, egui::vec2(node_radius * 2.0, node_radius * 2.0))
    }

    fn node_info(&self, model: &ModelInfo, labels: &[String], layer_idx: usize, node_idx: usize) -> String {
        let mut s = if layer_idx == 0 {
            "Input node".to_string()
        } else {
            let l = &model.layers[layer_idx - 1];
            let mut s = format!("Layer: {}\n", labels[layer_idx]);
            s.push_str(&format!("Type: {}\n", l.layer_type));
            if let Some(activation) = &l.activation {
                s.push_str(&format!("Activation: {}\n", activation));
            }
            if let Some(input_dim) = l.input_dim {
                s.push_str(&format!("Input dim: {}\n", input_dim));
            }
            if let Some(output_dim) = l.output_dim {
                s.push_str(&format!("Output dim: {}\n", output_dim));
            }
            s
        };
        // Add activation value if available
        if let Some(ref acts) = self.activations {
            if let Some(layer_acts) = acts.get(layer_idx) {
                if let Some(&act) = layer_acts.get(node_idx) {
                    s.push_str(&format!("Activation value: {:.4}\n", act));
                }
            }
        }
        s
    }

    fn draw_input_ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label("Input:");
            for v in &mut self.input_values {
                ui.add(egui::DragValue::new(v).speed(0.05));
            }
        });
    }

    fn draw_network(&mut self, ui: &mut egui::Ui) {
        let rect = ui.available_rect_before_wrap();
        // Mouse wheel zoom
        if let Some(pos) = ui.input(|i| i.pointer.hover_pos()) {
            if rect.contains(pos) {
                let scroll = ui.input(|i| i.raw_scroll_delta.y);
                if scroll != 0.0 {
                    // Continuous zoom: multiply by a factor per scroll step
                    let zoom_factor = 1.15_f32;
                    let old_zoom = self.zoom;
                    let new_zoom = (self.zoom * zoom_factor.powf(scroll.signum())).clamp(0.2, 5.0);
                    let before = (pos - self.pan) / old_zoom;
                    self.zoom = new_zoom;
                    let after = before * self.zoom;
                    self.pan += pos - (after + self.pan);
                }
            }
        }
        // Mouse drag pan
        let resp = ui.interact(rect, egui::Id::new("pan"), egui::Sense::drag());
        if resp.drag_started() { self.is_panning = true; self.last_pan_pos = resp.interact_pointer_pos(); }
        if self.is_panning && resp.dragged() {
            if let (Some(last), Some(cur)) = (self.last_pan_pos, resp.interact_pointer_pos()) {
                self.pan += cur - last;
                self.last_pan_pos = Some(cur);
            }
        }
        if resp.drag_stopped() { self.is_panning = false; self.last_pan_pos = None; }
        // Fetch weights before any immutable borrow of self
        if self.weights.is_none() {
            self.weights = self.call_get_weights();
        }
        if let Some(model) = &self.model_info {
            let n_layers = model.layers.len();
            if n_layers == 0 {
                ui.label("No layers to visualize.");
                return;
            }
            // Layout parameters
            let mut node_counts = Vec::with_capacity(n_layers + 1);
            let mut labels = Vec::with_capacity(n_layers + 1);
            let input_dim = model.layers.first().and_then(|l| l.input_dim).unwrap_or(1);
            node_counts.push(input_dim);
            labels.push("Input".to_string());
            for (i, layer) in model.layers.iter().enumerate() {
                let n = layer.output_dim.unwrap_or(1);
                node_counts.push(n);
                let is_last = i == n_layers - 1;
                let label = if is_last {
                    format!("{} (Output)", layer.layer_type)
                } else {
                    layer.layer_type.clone()
                };
                labels.push(label);
            }
            let n_columns = node_counts.len();
            // Save previous clip rect and set new one for the graph area
            let prev_clip = ui.clip_rect();
            ui.set_clip_rect(rect);
            let painter = ui.painter();
            // Draw a box around the graph area (manual, since StrokeKind is not available)
            let lt = rect.left_top();
            let rt = rect.right_top();
            let lb = rect.left_bottom();
            let rb = rect.right_bottom();
            let stroke = egui::Stroke::new(2.0, egui::Color32::DARK_GRAY);
            painter.line_segment([lt, rt], stroke);
            painter.line_segment([rt, rb], stroke);
            painter.line_segment([rb, lb], stroke);
            painter.line_segment([lb, lt], stroke);

            // Clip all graph drawing to the bounding rect
            let width = rect.width();
            let height = rect.height();
            let margin = 40.0;
            let _node_radius = 12.0;
            let max_nodes = *node_counts.iter().max().unwrap_or(&1);
            let min_node_spacing = 8.0; // Minimum vertical gap between nodes
            let node_spacing = ((height - 2.0 * margin) / (max_nodes as f32 + 1.0)).clamp(min_node_spacing, 40.0);
            let total_graph_height = (max_nodes as f32 + 1.0) * node_spacing;
            let y_offset = rect.top() + (height - total_graph_height) / 2.0;
            let fixed_layer_spacing = 120.0; // or any value you prefer
            let _layer_spacing = fixed_layer_spacing;
            let _edge_inset = 60.0;
            // Dynamically compute horizontal gap between columns based on node counts
            let mut col_gaps = Vec::with_capacity(n_columns - 1);
            let base_gap = 120.0;
            for i in 0..n_columns - 1 {
                let n1 = node_counts[i] as f32;
                let n2 = node_counts[i + 1] as f32;
                // Increase gap for larger columns, scale with sqrt for visual balance
                let gap = base_gap + 8.0 * ((n1.max(n2)).sqrt());
                col_gaps.push(gap);
            }
            let total_graph_width: f32 = col_gaps.iter().sum();
            let x_offset = rect.left() + (width - total_graph_width) / 2.0;
            let mut layer_xs = Vec::with_capacity(n_columns);
            let mut x = x_offset;
            layer_xs.push(x);
            for gap in &col_gaps {
                x += *gap;
                layer_xs.push(x);
            }
            // Node positions
            let mut node_positions: Vec<Vec<egui::Pos2>> = Vec::with_capacity(n_columns);
            for (layer_idx, &n_nodes) in node_counts.iter().enumerate() {
                let mut positions = Vec::with_capacity(n_nodes);
                let layer_height = (n_nodes as f32 + 1.0) * node_spacing;
                let layer_y_offset = y_offset + (total_graph_height - layer_height) / 2.0;
                for node_idx in 0..n_nodes {
                    let y = layer_y_offset + (node_idx as f32 + 1.0) * node_spacing;
                    let pos = (egui::pos2(layer_xs[layer_idx], y) * self.zoom) + self.pan;
                    positions.push(pos);
                }
                node_positions.push(positions);
            }
            // Draw connections with weight-based thickness and color
            for l in 0..(n_columns - 1) {
                let w_layer = self.weights.as_ref().and_then(|w| w.get(l));
                for (from_idx, &from) in node_positions[l].iter().enumerate() {
                    for (to_idx, &to) in node_positions[l + 1].iter().enumerate() {
                        let weight = w_layer.and_then(|w| w.get(from_idx)).and_then(|v| v.get(to_idx)).copied().unwrap_or(0.0);
                        // Normalize weight for thickness: map [-max_abs, max_abs] to [2.0, 10.0]
                        let max_abs = 1.0; // Lower for more visible thickness
                        let abs_w = (weight.abs() / max_abs).min(1.0);
                        let thickness = 2.0 + 8.0 * abs_w;
                        let color = egui::Color32::from_rgb(120, 120, 220); // Constant color
                        let _response = painter.line_segment([
                            from,
                            to
                        ], egui::Stroke::new(thickness, color));
                        // Edge selection: check if mouse is near the edge
                        if ui.input(|i| i.pointer.any_click()) {
                            let pointer_pos = ui.input(|i| i.pointer.interact_pos());
                            if let Some(pos) = pointer_pos {
                                let dist = distance_to_segment(pos, from, to);
                                if dist < 8.0 {
                                    self.selected_edge = Some((l, from_idx, to_idx));
                                    self.selected_node = None; // Deselect node if edge is selected
                                }
                            }
                        }
                    }
                }
            }
            // Draw nodes and handle interactivity
            for (layer_idx, positions) in node_positions.iter().enumerate() {
                for (node_idx, &pos) in positions.iter().enumerate() {
                    let rect = self.node_rect(pos, _node_radius * self.zoom); // scale node radius
                    let node_id = egui::Id::new(("node", layer_idx, node_idx));
                    if ui.interact(rect, node_id, egui::Sense::click()).clicked() {
                        if self.selected_node == Some((layer_idx, node_idx)) {
                            self.selected_node = None;
                        } else {
                            self.selected_node = Some((layer_idx, node_idx));
                            self.selected_edge = None;
                        }
                    }
                    let color = if let Some(ref acts) = self.activations {
                        if let Some(layer_acts) = acts.get(layer_idx) {
                            if let Some(&act) = layer_acts.get(node_idx) {
                                let v = (act * 255.0).clamp(0.0, 255.0) as u8;
                                egui::Color32::from_rgb(v, 255 - v, 100)
                            } else {
                                self.node_color(layer_idx, node_idx)
                            }
                        } else {
                            self.node_color(layer_idx, node_idx)
                        }
                    } else {
                        self.node_color(layer_idx, node_idx)
                    };
                    painter.circle_filled(pos, _node_radius, color);
                }
            }
            // Draw labels (centered, high-contrast)
            for (layer_idx, &label_x) in layer_xs.iter().enumerate() {
                let y = y_offset + margin / 2.0;
                // Apply zoom and pan to label positions
                let pos = (egui::pos2(label_x, y) * self.zoom) + self.pan;
                painter.text(
                    pos,
                    egui::Align2::CENTER_CENTER,
                    &labels[layer_idx],
                    egui::FontId::proportional(14.0),
                    egui::Color32::WHITE,
                );
            }
            // Info box for selected node
            if let Some((layer_idx, node_idx)) = self.selected_node {
                let pos = node_positions[layer_idx][node_idx];
                let info = self.node_info(model, &labels, layer_idx, node_idx);
                let lines: Vec<&str> = info.lines().collect();
                let line_height = 16.0;
                let vertical_padding = 16.0;
                let horizontal_padding = 16.0;
                // Calculate max line width in pixels
                let mut max_line_width: f32 = 0.0;
                for line in &lines {
                    let galley = ui.painter().layout_no_wrap(
                        line.to_string(),
                        egui::FontId::proportional(13.0),
                        egui::Color32::BLACK,
                    );
                    max_line_width = max_line_width.max(galley.size().x);
                }
                let box_width = max_line_width + 2.0 * horizontal_padding;
                let box_height = lines.len() as f32 * line_height + 2.0 * vertical_padding;
                let box_size = egui::vec2(box_width, box_height);
                let margin = 10.0;
                // Default offset: right of node
                let mut box_pos = pos + egui::vec2(30.0, -40.0);
                let min = rect.left_top();
                let max = rect.right_bottom() - box_size;
                // If box would go off right edge, flip to left
                if box_pos.x + box_size.x > max.x {
                    box_pos.x = pos.x - box_size.x - 30.0;
                }
                // If box would go off left edge, clamp to left
                if box_pos.x < min.x {
                    box_pos.x = min.x + margin;
                }
                // If box would go off bottom, move up
                if box_pos.y + box_size.y > max.y {
                    box_pos.y = max.y - margin;
                }
                // If box would go off top, clamp to top
                if box_pos.y < min.y {
                    box_pos.y = min.y + margin;
                }
                let rect_box = egui::Rect::from_min_size(box_pos, box_size);
                painter.rect_filled(rect_box, 8.0, egui::Color32::from_rgba_unmultiplied(255,255,220,230));
                // Draw info text, vertically centered with extra padding
                let total_text_height = lines.len() as f32 * line_height;
                let start_y = rect_box.top() + vertical_padding + (rect_box.height() - 2.0 * vertical_padding - total_text_height) / 2.0 + line_height / 2.0;
                for (i, line) in lines.iter().enumerate() {
                    let text_pos = egui::pos2(rect_box.center().x, start_y + i as f32 * line_height);
                    painter.text(
                        text_pos,
                        egui::Align2::CENTER_CENTER,
                        *line,
                        egui::FontId::proportional(13.0),
                        egui::Color32::BLACK,
                    );
                }
            }
            // Info box for selected edge
            if let Some((layer, from_idx, to_idx)) = self.selected_edge {
                let from = node_positions[layer][from_idx];
                let to = node_positions[layer + 1][to_idx];
                let mid = egui::pos2((from.x + to.x) / 2.0, (from.y + to.y) / 2.0);
                let weight = self.weights.as_ref()
                    .and_then(|w| w.get(layer))
                    .and_then(|w| w.get(from_idx))
                    .and_then(|v| v.get(to_idx))
                    .copied()
                    .unwrap_or(0.0);
                let info = format!("Edge\nLayer: {}\nFrom: {}\nTo: {}\nWeight: {:.4}", layer, from_idx, to_idx, weight);
                let lines: Vec<&str> = info.lines().collect();
                let line_height = 16.0;
                let vertical_padding = 16.0;
                let horizontal_padding = 16.0;
                let mut max_line_width: f32 = 0.0;
                for line in &lines {
                    let galley = ui.painter().layout_no_wrap(
                        line.to_string(),
                        egui::FontId::proportional(13.0),
                        egui::Color32::BLACK,
                    );
                    max_line_width = max_line_width.max(galley.size().x);
                }
                let box_width = max_line_width + 2.0 * horizontal_padding;
                let box_height = lines.len() as f32 * line_height + 2.0 * vertical_padding;
                let box_size = egui::vec2(box_width, box_height);
                let margin = 10.0;
                let mut box_pos = mid + egui::vec2(30.0, -40.0);
                let min = rect.left_top();
                let max = rect.right_bottom() - box_size;
                if box_pos.x + box_size.x > max.x {
                    box_pos.x = mid.x - box_size.x - 30.0;
                }
                if box_pos.x < min.x {
                    box_pos.x = min.x + margin;
                }
                if box_pos.y + box_size.y > max.y {
                    box_pos.y = max.y - margin;
                }
                if box_pos.y < min.y {
                    box_pos.y = min.y + margin;
                }
                let rect_box = egui::Rect::from_min_size(box_pos, box_size);
                painter.rect_filled(rect_box, 8.0, egui::Color32::from_rgba_unmultiplied(255,255,220,230));
                let total_text_height = lines.len() as f32 * line_height;
                let start_y = rect_box.top() + vertical_padding + (rect_box.height() - 2.0 * vertical_padding - total_text_height) / 2.0 + line_height / 2.0;
                for (i, line) in lines.iter().enumerate() {
                    let text_pos = egui::pos2(rect_box.center().x, start_y + i as f32 * line_height);
                    painter.text(
                        text_pos,
                        egui::Align2::CENTER_CENTER,
                        *line,
                        egui::FontId::proportional(13.0),
                        egui::Color32::BLACK,
                    );
                }
            }
            // Restore previous clip rect
            ui.set_clip_rect(prev_clip);
        } else {
            ui.label("No model loaded. Click 'Call Python NN' to load.");
        }
    }

    fn draw_model_editor(&mut self, ui: &mut egui::Ui) {
        ui.heading("Edit Neural Network Structure");
        ui.horizontal(|ui| {
            ui.label("Input nodes:");
            ui.add(egui::DragValue::new(&mut self.editable_input_dim).range(1..=16));
        });
        if ui.button("Add Hidden Layer").clicked() {
            let prev_output = self.editable_layers.last().and_then(|l| l.output_dim).unwrap_or(1);
            self.editable_layers.insert(
                self.editable_layers.len() - 1,
                LayerInfo {
                    layer_type: "Dense".to_string(),
                    input_dim: Some(prev_output),
                    output_dim: Some(3),
                    activation: Some("relu".to_string()),
                },
            );
        }
        let mut remove_indices = Vec::new();
        for i in 0..self.editable_layers.len() {
            let is_output = i == self.editable_layers.len() - 1;
            let is_last = is_output; // store before mutable borrow
            let layer = &mut self.editable_layers[i];
            let header_label = if is_output {
                format!("Output Layer")
            } else {
                format!("Hidden Layer {}", i + 1)
            };
            egui::CollapsingHeader::new(header_label)
                .default_open(is_last)
                .show(ui, |ui| {
                    if !is_output {
                        ui.horizontal(|ui| {
                            if ui.button("Remove").clicked() {
                                remove_indices.push(i);
                            }
                        });
                        ui.horizontal(|ui| {
                            ui.label("Nodes:");
                            if let Some(ref mut out_dim) = layer.output_dim {
                                ui.add(egui::DragValue::new(out_dim).range(1..=32));
                            }
                        });
                    } else {
                        ui.horizontal(|ui| {
                            ui.label("Nodes:");
                            if let Some(ref mut out_dim) = layer.output_dim {
                                ui.add(egui::DragValue::new(out_dim).range(1..=16));
                            }
                        });
                    }
                    ui.horizontal(|ui| {
                        ui.label("Activation:");
                        let activations = ["relu", "sigmoid", "tanh", "none"];
                        let mut act = layer.activation.clone().unwrap_or("none".to_string());
                        egui::ComboBox::from_id_salt(format!("act_{}_{}", i, is_output)).selected_text(&act)
                            .show_ui(ui, |ui| {
                                for a in activations.iter() {
                                    ui.selectable_value(&mut act, a.to_string(), *a);
                                }
                            });
                        layer.activation = if act == "none" { None } else { Some(act) };
                    });
                });
        }
        // Remove layers after the loop to avoid borrow checker issues
        for &idx in remove_indices.iter().rev() {
            self.editable_layers.remove(idx);
        }
        if ui.button("Apply Changes").clicked() {
            self.reload_rust_nn();
        }
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.label(&self.python_result);
            ui.separator();
            egui::ScrollArea::vertical().show(ui, |ui| {
                self.draw_model_editor(ui);
                // --- Controls above the graph ---
                ui.horizontal(|ui| {
                    if ui.button("-").clicked() { self.zoom = (self.zoom * 0.9).max(0.2); }
                    if ui.button("+").clicked() { self.zoom = (self.zoom * 1.1).min(5.0); }
                    ui.label(format!("Zoom: {:.2}x", self.zoom));
                    if ui.button("Reset View").clicked() { self.zoom = 1.0; self.pan = egui::Vec2::ZERO; }
                });
                self.draw_input_ui(ui);
                if ui.button("Run Forward Pass").clicked() {
                    self.activations = self.call_forward_pass().map(|fp| fp.activations);
                }
                // --- End controls ---
                self.draw_network(ui);
            });
        });
    }
}

// Helper function to compute distance from a point to a line segment
fn distance_to_segment(p: egui::Pos2, a: egui::Pos2, b: egui::Pos2) -> f32 {
    let ab = b - a;
    let ap = p - a;
    let ab_len = ab.length_sq();
    if ab_len == 0.0 {
        return ap.length();
    }
    let t = (ap.x * ab.x + ap.y * ab.y) / ab_len;
    let t = t.clamp(0.0, 1.0);
    let proj = a + ab * t;
    (p - proj).length()
}

fn main() {
    let options = eframe::NativeOptions::default();
    let _ = eframe::run_native(
        "NN Visualizer",
        options,
        Box::new(|_cc| Ok(Box::new(MyApp::default()))),
    );
}
