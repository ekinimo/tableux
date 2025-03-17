mod common;

use svg::node::element::{Circle, Group, Line, Path, Rectangle, Text};
use svg::node::Text as TextContent;
use svg::Document;
use wasm_bindgen::prelude::*;

use common::{parse_lisp, FormulaParser, Rule, Tableux, TableuxIdx};

use pest::Parser;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

#[wasm_bindgen]
pub struct TableuxVisualizer {
    tableux: Tableux,
}

#[wasm_bindgen]
impl TableuxVisualizer {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        TableuxVisualizer {
            tableux: Tableux::default(),
        }
    }

    // This function ONLY parses and sets up the formula, but does not run any steps
    pub fn parse_formula(&mut self, formula_str: &str) -> Result<(), String> {
        self.tableux.clear();

        // Break up the parsing to avoid borrowing issues
        let parse_result = FormulaParser::parse(Rule::formula, formula_str);

        match parse_result {
            Ok(mut parsed) => {
                let parsed_inner = parsed.next().unwrap();
                let formula_inner = parsed_inner.into_inner().next().unwrap();

                // Parse the formula
                let formula = parse_lisp(&mut self.tableux.formulas, formula_inner);

                // Set the hypothesis
                self.tableux.parse_hypothesis(formula);

                Ok(())
            }
            Err(e) => Err(format!("Parse error: {}", e)),
        }
    }

    pub fn step(&mut self, steps: usize) -> bool {
        // Take only one or a few steps at a time
        let initial_open_count = self.tableux.open.len();
        let initial_cant_progress_count = self.tableux.cant_progress.len();

        if !self.tableux.open.is_empty() {
            for _ in 0..steps {
                self.tableux.step();
                self.tableux.process_unprocessed_nodes();

                // If no more open branches, stop
                if self.tableux.open.is_empty() {
                    break;
                }
            }

            // Return true if we made progress
            return initial_open_count != self.tableux.open.len()
                || initial_cant_progress_count != self.tableux.cant_progress.len();
        }

        false
    }

    pub fn is_complete(&self) -> bool {
        self.tableux.open.is_empty()
    }

    pub fn is_tautology(&self) -> bool {
        self.tableux.open.is_empty() && self.tableux.cant_progress.is_empty()
    }

    #[wasm_bindgen]
    pub fn generate_svg(&self) -> String {
        self._generate_svg()
    }

    pub fn parse_and_prove(&mut self, formula_str: &str, steps: usize) -> String {
        match self.parse_formula(formula_str) {
            Ok(_) => {
                // Run the proof
                self.tableux.proove(steps);

                // Generate SVG after everything is done
                self._generate_svg()
            }
            Err(e) => e,
        }
    }

    fn _generate_svg(&self) -> String {
        // Constants for layout
        let min_node_width = 200; // Minimum node width
        let min_node_height = 100; // Minimum node height
        let h_spacing = 60; // Increased horizontal spacing between sibling nodes
        let v_spacing = 80; // Vertical spacing between parent and child
        let left_padding = 100; // Extra padding on the left side to prevent chopping

        // Pre-calculate node sizes based on content
        let mut node_sizes = std::collections::HashMap::new();
        for i in 0..self.tableux.elements.len() {
            let idx = TableuxIdx(i);
            let element = &self.tableux[idx];
            let formula = self.tableux.formulas.display(element.formula);

            // Calculate required size based on formula length
            let (width, height) = self.calculate_node_size(&formula);
            node_sizes.insert(idx, (width, height));
        }

        // Calculate tree layout
        let (positions, tree_width, tree_height) = self.calculate_tree_layout(
            min_node_width,
            min_node_height,
            h_spacing,
            v_spacing,
            &node_sizes,
            left_padding,
        );

        // Create SVG with size that fits the entire tree
        let mut document = Document::new()
            .set("width", tree_width)
            .set("height", tree_height)
            .set("viewBox", (0, 0, tree_width, tree_height))
            .set("style", "background-color: #000000;");

        // Add title
        let title = if self.tableux.open.is_empty() && self.tableux.cant_progress.is_empty() {
            "TAUTOLOGY"
        } else if self.tableux.open.is_empty() {
            "NOT COMPLETE"
        } else {
            "PROOF IN PROGRESS"
        };

        document = document.add(
            Text::new()
                .set("x", 20)
                .set("y", 30)
                .set("font-family", "Arial")
                .set("font-size", 15)
                .set("font-weight", "bold")
                .set("fill", "white")
                .add(TextContent::new(title)),
        );

        if self.tableux.elements.is_empty() {
            // If no nodes, add a message
            document = document.add(
                Text::new()
                    .set("x", tree_width / 2)
                    .set("y", tree_height / 2)
                    .set("text-anchor", "middle")
                    .set("font-family", "Arial")
                    .set("font-size", 20)
                    .set("fill", "white")
                    .add(TextContent::new("No tableaux to display")),
            );
            return document.to_string();
        }

        // Draw the edges first
        for (idx, (left, right)) in &self.tableux.children {
            if let Some(parent_pos) = positions.get(idx) {
                let parent_size = *node_sizes
                    .get(idx)
                    .unwrap_or(&(min_node_width, min_node_height));
                let parent_width = parent_size.0;
                let parent_height = parent_size.1;

                if let Some(left_idx) = left {
                    if let Some(left_pos) = positions.get(left_idx) {
                        let left_size = *node_sizes
                            .get(left_idx)
                            .unwrap_or(&(min_node_width, min_node_height));
                        let left_height = left_size.1;

                        document = document.add(
                            Line::new()
                                .set("x1", parent_pos.0)
                                .set("y1", parent_pos.1 + parent_height / 2)
                                .set("x2", left_pos.0)
                                .set("y2", left_pos.1 - left_height / 2)
                                .set("stroke", "#dddddd") // Light gray instead of black
                                .set("stroke-width", 2),
                        );
                    }
                }

                if let Some(right_idx) = right {
                    if let Some(right_pos) = positions.get(right_idx) {
                        let right_size = *node_sizes
                            .get(right_idx)
                            .unwrap_or(&(min_node_width, min_node_height));
                        let right_height = right_size.1;

                        document = document.add(
                            Line::new()
                                .set("x1", parent_pos.0)
                                .set("y1", parent_pos.1 + parent_height / 2)
                                .set("x2", right_pos.0)
                                .set("y2", right_pos.1 - right_height / 2)
                                .set("stroke", "#dddddd") // Light gray instead of black
                                .set("stroke-width", 2),
                        );
                    }
                }
            }
        }

        // Create a map for closed branch connections
        let mut closures = Vec::new();
        for (closed_idx, contradiction_idx) in &self.tableux.closed_branches {
            if let (Some(closed_pos), Some(contra_pos)) =
                (positions.get(closed_idx), positions.get(contradiction_idx))
            {
                closures.push((*closed_idx, *contradiction_idx, *closed_pos, *contra_pos));
            }
        }

        // Draw closed branch connections using curves to avoid node occlusion
        for (closed_idx, contradiction_idx, closed_pos, contra_pos) in closures {
            // Get node sizes
            let closed_size = *node_sizes
                .get(&closed_idx)
                .unwrap_or(&(min_node_width, min_node_height));
            let contra_size = *node_sizes
                .get(&contradiction_idx)
                .unwrap_or(&(min_node_width, min_node_height));
            let closed_width = closed_size.0;
            let contra_width = contra_size.0;

            // Generate a unique color for this closure
            let color_seed = (closed_idx.0 + contradiction_idx.0) % 12;
            let colors = [
                "#e57373", "#f06292", "#ba68c8", "#9575cd", "#7986cb", "#64b5f6", "#4fc3f7",
                "#4dd0e1", "#4db6ac", "#81c784", "#aed581", "#fff176",
            ];
            let color = colors[color_seed];

            // Determine which side to connect based on relative positions
            let (start_x, start_y, end_x, end_y);

            if closed_pos.0 <= contra_pos.0 {
                // Connect right side of closed to left side of contra
                start_x = closed_pos.0 + closed_width / 2;
                start_y = closed_pos.1;
                end_x = contra_pos.0 - contra_width / 2;
                end_y = contra_pos.1;
            } else {
                // Connect left side of closed to right side of contra
                start_x = closed_pos.0 - closed_width / 2;
                start_y = closed_pos.1;
                end_x = contra_pos.0 + contra_width / 2;
                end_y = contra_pos.1;
            }

            // Create control points for a curve that routes around nodes
            let dx = end_x - start_x;
            let dy = end_y - start_y;
            let distance = ((dx * dx + dy * dy) as f64).sqrt() as i32;

            // Choose control point direction based on relative positions
            let ctrl_x_offset = if closed_pos.0 <= contra_pos.0 {
                distance / 4 // Curve outward/right
            } else {
                -distance / 4 // Curve outward/left
            };

            let ctrl1_x = start_x + ctrl_x_offset;
            let ctrl1_y = start_y;
            let ctrl2_x = end_x - ctrl_x_offset;
            let ctrl2_y = end_y;

            // Draw curved dashed line for contradiction
            let path_data = format!(
                "M {} {} C {} {}, {} {}, {} {}",
                start_x, start_y, ctrl1_x, ctrl1_y, ctrl2_x, ctrl2_y, end_x, end_y
            );

            document = document.add(
                Path::new()
                    .set("d", path_data)
                    .set("stroke", color)
                    .set("stroke-width", 2)
                    .set("stroke-dasharray", "5,5")
                    .set("fill", "none"),
            );

            // Add circles with node indexes at both ends of the path
            document = document.add(
                Circle::new()
                    .set("cx", start_x)
                    .set("cy", start_y)
                    .set("r", 12)
                    .set("fill", color),
            );

            document = document.add(
                Text::new()
                    .set("x", start_x)
                    .set("y", start_y + 4)
                    .set("text-anchor", "middle")
                    .set("font-family", "Arial")
                    .set("font-size", 10)
                    .set("fill", "white")
                    .add(TextContent::new(format!("{}", closed_idx.0))),
            );

            document = document.add(
                Circle::new()
                    .set("cx", end_x)
                    .set("cy", end_y)
                    .set("r", 12)
                    .set("fill", color),
            );

            document = document.add(
                Text::new()
                    .set("x", end_x)
                    .set("y", end_y + 4)
                    .set("text-anchor", "middle")
                    .set("font-family", "Arial")
                    .set("font-size", 10)
                    .set("fill", "white")
                    .add(TextContent::new(format!("{}", contradiction_idx.0))),
            );
        }

        // Draw the nodes
        for (idx, pos) in &positions {
            let element = &self.tableux[*idx];
            let node_size = *node_sizes
                .get(idx)
                .unwrap_or(&(min_node_width, min_node_height));
            let node_width = node_size.0;
            let node_height = node_size.1;

            // Node background color
            let is_closed = self.tableux.closed_branches.iter().any(|(a, _)| a == idx);
            let bg_color = if is_closed {
                "#ffcccc" // Closed branch - light red
            } else if self.tableux.open.contains(idx) {
                "#ccffcc" // Open branch - light green
            } else if self.tableux.cant_progress.contains(idx) {
                "#ffffcc" // Can't progress - light yellow
            } else {
                "#ffffff" // Normal - white
            };

            // Create node
            let mut node = Group::new();

            // Add rectangle
            node = node.add(
                Rectangle::new()
                    .set("x", pos.0 - node_width / 2)
                    .set("y", pos.1 - node_height / 2)
                    .set("width", node_width)
                    .set("height", node_height)
                    .set("rx", 10)
                    .set("ry", 10)
                    .set("fill", bg_color)
                    .set("stroke", "black")
                    .set("stroke-width", 2),
            );

            // Add index number
            node = node.add(
                Text::new()
                    .set("x", pos.0 - node_width / 2 + 10)
                    .set("y", pos.1 - node_height / 2 + 20)
                    .set("font-family", "monospace")
                    .set("font-size", 12)
                    .set("fill", "#666666")
                    .add(TextContent::new(format!("[{}]", idx.0))),
            );

            // Add sign (T/F) on the LEFT side
            node = node.add(
                Text::new()
                    .set("x", pos.0 - node_width / 2 + 25)
                    .set("y", pos.1)
                    .set("font-family", "Arial")
                    .set("font-size", 18)
                    .set("font-weight", "bold")
                    .set("dominant-baseline", "middle")
                    .add(TextContent::new(if element.sign { "T" } else { "F" })),
            );

            // Add formula
            let formula = self.tableux.formulas.display(element.formula);

            // Calculate appropriate text size for fitting
            let text_size = if formula.len() > 100 {
                11 // Small text for very long formulas
            } else if formula.len() > 50 {
                12 // Medium text for long formulas
            } else {
                13 // Regular text size
            };

            // Determine max chars per line based on node width
            let max_chars = (node_width / 10) as usize;
            let wrap_text = self.wrap_text(&formula, max_chars);

            let line_height = (text_size as i32) + 6; // Spacing between lines
            let start_y = pos.1 - ((wrap_text.len() as i32 * line_height) / 2) + 15;

            for (i, line) in wrap_text.iter().enumerate() {
                node = node.add(
                    Text::new()
                        .set("x", pos.0 - node_width / 2 + 50) // Left-aligned with padding
                        .set("y", start_y + (i as i32 * line_height))
                        .set("text-anchor", "start")
                        .set("font-family", "monospace")
                        .set("font-size", text_size)
                        .add(TextContent::new(line)),
                );
            }

            // Add closure indicator if needed
            if is_closed {
                let closed_with = self
                    .tableux
                    .closed_branches
                    .iter()
                    .find(|(a, _)| a == idx)
                    .map(|(_, b)| *b)
                    .unwrap();

                // Find the color for this closure
                let color_seed = (idx.0 + closed_with.0) % 12;
                let colors = [
                    "#e57373", "#f06292", "#ba68c8", "#9575cd", "#7986cb", "#64b5f6", "#4fc3f7",
                    "#4dd0e1", "#4db6ac", "#81c784", "#aed581", "#fff176",
                ];
                let color = colors[color_seed];

                // Add indicator in top-right
                let badge_x = pos.0 + node_width / 2 - 20;
                let badge_y = pos.1 - node_height / 2 + 20;

                // Circle
                node = node.add(
                    Circle::new()
                        .set("cx", badge_x)
                        .set("cy", badge_y)
                        .set("r", 15)
                        .set("fill", color),
                );

                // X mark
                node = node.add(
                    Text::new()
                        .set("x", badge_x)
                        .set("y", badge_y + 5)
                        .set("text-anchor", "middle")
                        .set("font-family", "Arial")
                        .set("font-size", 14)
                        .set("fill", "white")
                        .add(TextContent::new("×")),
                );

                // Text showing which node closed it
                node = node.add(
                    Text::new()
                        .set("x", badge_x - 20)
                        .set("y", badge_y)
                        .set("text-anchor", "end")
                        .set("dominant-baseline", "middle")
                        .set("font-family", "Arial")
                        .set("font-size", 12)
                        .set("fill", color)
                        .set("font-weight", "bold")
                        .add(TextContent::new(format!("[{}]", closed_with.0))),
                );
            }

            document = document.add(node);
        }

        document.to_string()
    }

    // Calculate the tree layout with proper centering
    fn calculate_tree_layout(
        &self,
        min_node_width: i32,
        min_node_height: i32,
        h_spacing: i32,
        v_spacing: i32,
        node_sizes: &std::collections::HashMap<TableuxIdx, (i32, i32)>,
        left_padding: i32,
    ) -> (std::collections::HashMap<TableuxIdx, (i32, i32)>, i32, i32) {
        // This function returns (positions, width, height)

        // Create a map to store positions (x, y)
        let mut positions = std::collections::HashMap::new();
        let mut max_width = 800;
        let mut max_height = 600;

        if self.tableux.elements.is_empty() {
            return (positions, max_width, max_height);
        }

        // Simple approach: assign initial positions
        self.assign_initial_positions(
            &mut positions,
            TableuxIdx(0),
            0,
            max_width / 2,
            80,
            max_width - 2 * left_padding,
            node_sizes,
            min_node_width,
            min_node_height,
            h_spacing,
            v_spacing,
        );

        // Find minimum x coordinate to calculate offset for centering
        let mut min_x = std::i32::MAX;
        for (idx, (x, _)) in &positions {
            let (node_width, _) = *node_sizes
                .get(idx)
                .unwrap_or(&(min_node_width, min_node_height));
            min_x = min_x.min(x - node_width / 2);
        }

        // Calculate offset to ensure all nodes are fully visible
        let offset = left_padding - min_x;

        // Apply offset to center the tree and find maximum bounds
        let mut adjusted_positions = std::collections::HashMap::new();
        for (idx, (x, y)) in &positions {
            let adjusted_x = x + offset;
            adjusted_positions.insert(*idx, (adjusted_x, *y));

            let (node_width, node_height) = *node_sizes
                .get(idx)
                .unwrap_or(&(min_node_width, min_node_height));
            max_width = max_width.max(adjusted_x + node_width / 2 + left_padding);
            max_height = max_height.max(y + node_height / 2 + 50);
        }

        (adjusted_positions, max_width, max_height)
    }

    // Recursively assign initial positions to nodes
    fn assign_initial_positions(
        &self,
        positions: &mut std::collections::HashMap<TableuxIdx, (i32, i32)>,
        idx: TableuxIdx,
        level: i32,
        x: i32,
        y: i32,
        available_width: i32,
        node_sizes: &std::collections::HashMap<TableuxIdx, (i32, i32)>,
        min_node_width: i32,
        min_node_height: i32,
        h_spacing: i32,
        v_spacing: i32,
    ) {
        // Store this node's position
        positions.insert(idx, (x, y));

        // Process children
        if let Some((left, right)) = self.tableux.children.get(&idx) {
            let (node_width, node_height) = *node_sizes
                .get(&idx)
                .unwrap_or(&(min_node_width, min_node_height));
            let next_y = y + node_height + v_spacing;

            match (left, right) {
                (Some(left_idx), Some(right_idx)) => {
                    // Binary branch - need to position both children with enough spacing
                    let left_size = *node_sizes
                        .get(left_idx)
                        .unwrap_or(&(min_node_width, min_node_height));
                    let right_size = *node_sizes
                        .get(right_idx)
                        .unwrap_or(&(min_node_width, min_node_height));
                    let left_width = left_size.0;
                    let right_width = right_size.0;

                    // Calculate horizontal spacing needed
                    let min_spacing = (left_width + right_width) / 2 + h_spacing;
                    let half_width = min_spacing.max(available_width / 3);

                    // Position left child
                    self.assign_initial_positions(
                        positions,
                        *left_idx,
                        level + 1,
                        x - half_width,
                        next_y,
                        available_width / 2,
                        node_sizes,
                        min_node_width,
                        min_node_height,
                        h_spacing,
                        v_spacing,
                    );

                    // Position right child
                    self.assign_initial_positions(
                        positions,
                        *right_idx,
                        level + 1,
                        x + half_width,
                        next_y,
                        available_width / 2,
                        node_sizes,
                        min_node_width,
                        min_node_height,
                        h_spacing,
                        v_spacing,
                    );
                }
                (Some(child_idx), None) | (None, Some(child_idx)) => {
                    // Single child - center under parent
                    self.assign_initial_positions(
                        positions,
                        *child_idx,
                        level + 1,
                        x,
                        next_y,
                        available_width,
                        node_sizes,
                        min_node_width,
                        min_node_height,
                        h_spacing,
                        v_spacing,
                    );
                }
                _ => {}
            }
        }
    }

    // Helper function to calculate node size based on text content
    fn calculate_node_size(&self, formula: &str) -> (i32, i32) {
        // Minimum node size
        let min_width = 200;
        let min_height = 100;

        // Estimate text dimensions
        let lines = self.wrap_text(formula, 20); // 20 chars per line
        let line_count = lines.len() as i32;

        // Estimate width based on longest line
        let max_line_len = lines.iter().map(|line| line.len()).max().unwrap_or(0) as i32;
        let char_width = 8; // Approximate width per character

        // Calculate dimensions with some padding
        let width = (max_line_len * char_width + 80).max(min_width); // +80 for T/F sign and padding
        let height = (line_count * 20 + 40).max(min_height); // +40 for padding

        (width, height)
    }

    // Helper function for text wrapping
    fn wrap_text(&self, text: &str, max_width: usize) -> Vec<String> {
        let mut lines = Vec::new();
        let mut current_line = String::new();

        // Pre-process the text to add spaces around symbols
        let processed = text
            .replace("(", " ( ")
            .replace(")", " ) ")
            .replace("∧", " ∧ ")
            .replace("∨", " ∨ ")
            .replace("¬", " ¬ ")
            .replace("⇒", " ⇒ ");

        for word in processed.split_whitespace() {
            if current_line.len() + word.len() + 1 > max_width && !current_line.is_empty() {
                lines.push(current_line);
                current_line = word.to_string();
            } else {
                if !current_line.is_empty() {
                    current_line.push(' ');
                }
                current_line.push_str(word);
            }
        }

        if !current_line.is_empty() {
            lines.push(current_line);
        }

        // If no lines were created (e.g., single very long word), split by characters
        if lines.is_empty() {
            let chars: Vec<char> = text.chars().collect();
            let mut i = 0;
            while i < chars.len() {
                let end = std::cmp::min(i + max_width, chars.len());
                lines.push(chars[i..end].iter().collect());
                i = end;
            }
        }

        lines
    }
}
