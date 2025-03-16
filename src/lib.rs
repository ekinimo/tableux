mod common;

use svg::node::element::{Circle, Group, Line, Path, Rectangle, Text};
use svg::node::Text as TextContent;
use svg::Document;
use wasm_bindgen::prelude::*;

use common::{
    parse_lisp, Formula, FormulaIdx, FormulaParser, FormulaPool, Rule, Tableux, TableuxIdx,
};

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
        // SVG dimensions
        let width = 1600;
        let height = 1200;
        let node_width = 400; // Reduced width
        let node_height = 150; // Reduced height

        // Create SVG document
        let mut document = Document::new()
            .set("width", width)
            .set("height", height)
            .set("viewBox", (0, 0, width, height))
            .set("style", "background-color: #fafafa;");

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
                .add(TextContent::new(title)),
        );

        // Create a map of node positions
        let mut positions = std::collections::HashMap::new();

        // Check if there are any nodes to render
        if !self.tableux.elements.is_empty() {
            self.calculate_positions(&mut positions, TableuxIdx(0), 0, width / 2, 80, width);

            // First, create a mapping of closed branch indicators
            let mut closed_branch_connections = Vec::new();
            for (closed_node, contradiction_node) in &self.tableux.closed_branches {
                if let (Some(pos_closed), Some(pos_contradiction)) = (
                    positions.get(closed_node),
                    positions.get(contradiction_node),
                ) {
                    closed_branch_connections.push((
                        *closed_node,
                        *contradiction_node,
                        *pos_closed,
                        *pos_contradiction,
                    ));
                }
            }

            // Draw edges first (so they're below nodes)
            for (idx, (left, right)) in &self.tableux.children {
                if let Some(pos_parent) = positions.get(idx) {
                    let (parent_x, parent_y) = *pos_parent;

                    if let Some(left_idx) = left {
                        if let Some(pos_left) = positions.get(left_idx) {
                            let (left_x, left_y) = *pos_left;
                            document = document.add(
                                Line::new()
                                    .set("x1", parent_x)
                                    .set("y1", parent_y + node_height / 2)
                                    .set("x2", left_x)
                                    .set("y2", left_y - node_height / 2)
                                    .set("stroke", "black")
                                    .set("stroke-width", 2),
                            );
                        }
                    }

                    if let Some(right_idx) = right {
                        if let Some(pos_right) = positions.get(right_idx) {
                            let (right_x, right_y) = *pos_right;
                            document = document.add(
                                Line::new()
                                    .set("x1", parent_x)
                                    .set("y1", parent_y + node_height / 2)
                                    .set("x2", right_x)
                                    .set("y2", right_y - node_height / 2)
                                    .set("stroke", "black")
                                    .set("stroke-width", 2),
                            );
                        }
                    }
                }
            }

            // Draw closed branch connections (curved lines showing contradiction relationships)
            for (closed_idx, contradiction_idx, pos_closed, pos_contradiction) in
                closed_branch_connections
            {
                let (closed_x, closed_y) = pos_closed;
                let (contra_x, contra_y) = pos_contradiction;

                // Generate a unique color for this closure
                let color_seed = (closed_idx.0 + contradiction_idx.0) % 12;
                let colors = [
                    "#e57373", "#f06292", "#ba68c8", "#9575cd", "#7986cb", "#64b5f6", "#4fc3f7",
                    "#4dd0e1", "#4db6ac", "#81c784", "#aed581", "#fff176",
                ];
                let color = colors[color_seed];

                // Draw curved connecting line
                let ctrl_x1 = closed_x - 100;
                let ctrl_y1 = closed_y + 50;
                let ctrl_x2 = contra_x - 100;
                let ctrl_y2 = contra_y - 50;

                // Bezier curve path
                let path_d = format!(
                    "M {} {} C {} {}, {} {}, {} {}",
                    closed_x - node_width / 4,
                    closed_y,
                    ctrl_x1,
                    ctrl_y1,
                    ctrl_x2,
                    ctrl_y2,
                    contra_x - node_width / 4,
                    contra_y
                );

                document = document.add(
                    Path::new()
                        .set("d", path_d)
                        .set("stroke", color)
                        .set("stroke-width", 3)
                        .set("stroke-dasharray", "5,5")
                        .set("fill", "none"),
                );

                // Add a triangle arrow manually at the end of the path
                let arrow_size = 7;
                let angle = (contra_y as f64 - ctrl_y2 as f64)
                    .atan2(contra_x as f64 - node_width as f64 / 4.0 - ctrl_x2 as f64);
                let arrow_x = contra_x as f64 - node_width as f64 / 4.0;
                let arrow_y = contra_y as f64;

                let point1_x =
                    arrow_x - arrow_size as f64 * angle.cos() - arrow_size as f64 * angle.sin();
                let point1_y =
                    arrow_y - arrow_size as f64 * angle.sin() + arrow_size as f64 * angle.cos();
                let point2_x = arrow_x;
                let point2_y = arrow_y;
                let point3_x =
                    arrow_x - arrow_size as f64 * angle.cos() + arrow_size as f64 * angle.sin();
                let point3_y =
                    arrow_y - arrow_size as f64 * angle.sin() - arrow_size as f64 * angle.cos();

                let arrow_path = format!(
                    "M {} {} L {} {} L {} {} Z",
                    point1_x, point1_y, point2_x, point2_y, point3_x, point3_y
                );

                document = document.add(
                    Path::new()
                        .set("d", arrow_path)
                        .set("fill", color)
                        .set("stroke", "none"),
                );

                // Add circles with node indexes at both ends of the path
                document = document.add(
                    Circle::new()
                        .set("cx", closed_x - node_width / 4)
                        .set("cy", closed_y)
                        .set("r", 12)
                        .set("fill", color),
                );

                document = document.add(
                    Text::new()
                        .set("x", closed_x - node_width / 4)
                        .set("y", closed_y + 4)
                        .set("text-anchor", "middle")
                        .set("font-family", "Arial")
                        .set("font-size", 10)
                        .set("fill", "white")
                        .add(TextContent::new(format!("{}", closed_idx.0))),
                );

                document = document.add(
                    Circle::new()
                        .set("cx", contra_x - node_width / 4)
                        .set("cy", contra_y)
                        .set("r", 12)
                        .set("fill", color),
                );

                document = document.add(
                    Text::new()
                        .set("x", contra_x - node_width / 4)
                        .set("y", contra_y + 4)
                        .set("text-anchor", "middle")
                        .set("font-family", "Arial")
                        .set("font-size", 10)
                        .set("fill", "white")
                        .add(TextContent::new(format!("{}", contradiction_idx.0))),
                );
            }

            // Draw nodes
            for (idx, pos) in &positions {
                let (x, y) = *pos;
                let element = &self.tableux[*idx];

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

                // Create node group
                let mut node = Group::new();

                // Add rectangle
                node = node.add(
                    Rectangle::new()
                        .set("x", x - node_width / 2)
                        .set("y", y - node_height / 2)
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
                        .set("x", x - node_width / 2 + 10)
                        .set("y", y - node_height / 2 + 20)
                        .set("font-family", "monospace")
                        .set("font-size", 12)
                        .set("fill", "#666666")
                        .add(TextContent::new(format!("[{}]", idx.0))),
                );

                // Add sign (T/F)
                node = node.add(
                    Text::new()
                        .set("x", x)
                        .set("y", y - node_height / 2 + 30)
                        .set("text-anchor", "middle")
                        .set("font-family", "Arial")
                        .set("font-size", 18)
                        .set("font-weight", "bold")
                        .add(TextContent::new(if element.sign { "T" } else { "F" })),
                );

                // Add formula with guaranteed full display
                let formula = self.tableux.formulas.display(element.formula);

                // Implementation of a word-wrapping algorithm
                let max_chars = 25; // Reduced max chars per line for smaller nodes
                let mut lines = Vec::new();

                // Split into lines with a simple character approach
                let words = formula.split_whitespace().collect::<Vec<_>>();
                let mut current_line = String::new();

                for word in words {
                    if current_line.len() + word.len() + 1 > max_chars && !current_line.is_empty() {
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

                // Handle case where we have no lines or single very long word
                if lines.is_empty() {
                    // Split by characters if needed
                    let chars: Vec<char> = formula.chars().collect();
                    let mut i = 0;
                    while i < chars.len() {
                        let end = std::cmp::min(i + max_chars, chars.len());
                        lines.push(chars[i..end].iter().collect());
                        i = end;
                    }
                }

                // Calculate vertical spacing
                let line_height = 22; // Reduced spacing between lines
                let start_y = y - ((lines.len() as i32 * line_height) / 2) + 15;

                // Add each line of the formula
                for (i, line) in lines.iter().enumerate() {
                    node = node.add(
                        Text::new()
                            .set("x", x)
                            .set("y", start_y + (i as i32 * line_height))
                            .set("text-anchor", "middle")
                            .set("font-family", "monospace")
                            .set("font-size", 13) // Smaller font size
                            .add(TextContent::new(line)),
                    );
                }

                document = document.add(node);

                // Add closure status badge if this is a closed branch
                if is_closed {
                    let closed_with = self
                        .tableux
                        .closed_branches
                        .iter()
                        .find(|(a, _)| a == idx)
                        .map(|(_, b)| b)
                        .unwrap();

                    // Find the color associated with this closure
                    let color_seed = (idx.0 + closed_with.0) % 12;
                    let colors = [
                        "#e57373", "#f06292", "#ba68c8", "#9575cd", "#7986cb", "#64b5f6",
                        "#4fc3f7", "#4dd0e1", "#4db6ac", "#81c784", "#aed581", "#fff176",
                    ];
                    let color = colors[color_seed];

                    document = document.add(
                        Circle::new()
                            .set("cx", x + node_width / 2 - 20)
                            .set("cy", y - node_height / 2 + 20)
                            .set("r", 15)
                            .set("fill", color),
                    );

                    document = document.add(
                        Text::new()
                            .set("x", x + node_width / 2 - 20)
                            .set("y", y - node_height / 2 + 25)
                            .set("text-anchor", "middle")
                            .set("font-family", "Arial")
                            .set("font-size", 13)
                            .set("fill", "white")
                            .add(TextContent::new("×")),
                    );
                }
            }
        } else {
            // If no nodes, add a message
            document = document.add(
                Text::new()
                    .set("x", width / 2)
                    .set("y", height / 2)
                    .set("text-anchor", "middle")
                    .set("font-family", "Arial")
                    .set("font-size", 20)
                    .add(TextContent::new("No tableaux to display")),
            );
        }

        document.to_string()
    }

    fn calculate_positions(
        &self,
        positions: &mut std::collections::HashMap<TableuxIdx, (i32, i32)>,
        idx: TableuxIdx,
        level: i32,
        x: i32,
        y: i32,
        width: i32,
    ) {
        positions.insert(idx, (x, y));

        if let Some((left, right)) = self.tableux.children.get(&idx) {
            let next_y = y + 180; // Reduced vertical spacing

            if let Some(left_idx) = left {
                let next_width = if right.is_some() { width / 2 } else { width };
                let left_x = if right.is_some() { x - width / 4 } else { x };
                self.calculate_positions(
                    positions,
                    *left_idx,
                    level + 1,
                    left_x,
                    next_y,
                    next_width,
                );
            }

            if let Some(right_idx) = right {
                let right_x = x + width / 4;
                self.calculate_positions(
                    positions,
                    *right_idx,
                    level + 1,
                    right_x,
                    next_y,
                    width / 2,
                );
            }
        }
    }
}
