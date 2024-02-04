use std::{cell::RefCell, rc::Rc};

use ecow::{EcoString, EcoVec};
use line_numbers::LinePositions;

use crate::{
    lexer::{CodeRange, RangeMode},
    parser::{AstNode, CasePatternKind},
};

use super::{block::Block, Opcode, Upvalue, Value, DEBUG};

#[derive(Debug, Clone)]
pub struct Local {
    name: EcoString,
    is_capture: bool,
    is_final: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct UpvalueRef {
    pub index: u8,
    pub is_local: bool,
}

#[derive(Debug)]
pub enum CompileError {
    #[allow(dead_code)]
    Failed,
}

#[derive(Debug, Clone)]
pub struct Closure {
    pub name: EcoString,
    pub arity: usize,
    pub chunk: Block,
    pub upvalues: Vec<Rc<Upvalue>>,
    pub upvalue_refs: Vec<UpvalueRef>,
}

impl PartialEq for Closure {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name && self.arity == other.arity
    }
}

type CResult = Result<(), CompileError>;

pub struct CompilerState {
    locals: Vec<Local>,
    scope_depth: usize,
    upvalue_refs: Vec<UpvalueRef>,

    lambda_id: usize,

    current_name: EcoString,
    current_id: usize,
}

impl CompilerState {
    pub fn new() -> Self {
        Self {
            locals: Vec::new(),
            scope_depth: 0,
            upvalue_refs: Vec::new(),

            lambda_id: 0,

            current_name: EcoString::new(),
            current_id: 0,
        }
    }
}

pub struct Compiler {
    states: Vec<CompilerState>,

    // TODO: Would this benefit from being an Rc?
    constants: Vec<Rc<Value>>,
    final_globals: Vec<usize>,

    line_positions: LinePositions,
    original_range: Option<CodeRange>,
}

impl Compiler {
    pub fn new(line_positions: LinePositions) -> Self {
        Self {
            states: vec![CompilerState::new()],

            constants: Vec::new(),
            final_globals: Vec::new(),

            line_positions,
            original_range: None,
        }
    }

    pub fn compile(&mut self, node: &AstNode, block: &mut Block) -> CResult {
        match node {
            AstNode::Prog { body, range } => {
                for (i, node) in body.iter().enumerate() {
                    self.compile(node, block)?;

                    if i < body.len() - 1 {
                        match node {
                            AstNode::ExprStmt { range, .. } => {
                                // Discard the result of every expression statement except
                                // the last one, which is the return value of the program.
                                // This is done so the stack doesn't get cluttered with unused
                                // values.
                                self.emit(block, Opcode::Pop.into(), range);
                            }
                            _ => {}
                        }
                    }
                }

                self.emit(block, Opcode::Return.into(), range);

                self.optimize(block);

                Ok(())
            }
            AstNode::Block { body, .. } => {
                self.states.last_mut().unwrap().scope_depth += 1;

                for node in body.iter() {
                    match node {
                        AstNode::ExprStmt { expr, .. } => match expr.as_ref() {
                            AstNode::BoolLit { .. }
                            | AstNode::StrLit { .. }
                            | AstNode::NumLit { .. } => {
                                if node != body.last().unwrap() {
                                    // If this is not the last expression in a block, its value is discarded
                                    println!("warning: unused value");
                                    continue;
                                }
                            }
                            _ => {}
                        },
                        _ => {}
                    }

                    self.compile(node, block)?;
                }

                let state = &mut self.states.last_mut().unwrap();
                state.scope_depth -= 1;

                Ok(())
            }
            AstNode::ExprStmt { expr, .. } => self.compile(expr, block),
            AstNode::FuncDecl {
                id,
                is_final,
                params,
                body,
                range,
            } => {
                let name = match id.as_ref() {
                    AstNode::Ident { name, .. } => name.clone(),
                    _ => unreachable!(),
                };

                if *is_final {
                    // Constant functions run once at the time of declaration, so that
                    // later calls will always point to the same value
                    self.compile(body, block)?;
                    self.emit_store(name, block, range, true);
                } else {
                    match body.as_ref() {
                        AstNode::Block { body, .. } => {
                            if body.len() == 1 {
                                match body.last().unwrap() {
                                    AstNode::BoolLit { .. }
                                    | AstNode::StrLit { .. }
                                    | AstNode::NumLit { .. } => {
                                        // TODO: Do this in a more robust way.
                                        // This could be done by marking every function as
                                        // impure by default, and then marking it as pure if it
                                        // doesn't contain any side effects, nor calls to impure
                                        // functions.
                                        // Pure functions with no parameters should then trigger
                                        // the warning.
                                        println!(
                                            "warning: function will always return the same value"
                                        );
                                        println!("         consider making it a constant");
                                    }
                                    _ => {}
                                }
                            }
                        }
                        _ => unreachable!(),
                    }

                    self.emit_closure(Some(&name), params, body, block, range)?;
                }

                Ok(())
            }
            AstNode::Lambda {
                params,
                body,
                range,
            } => self.emit_closure(None, params, body, block, range),
            AstNode::FuncRef { id, range } => {
                match id.as_ref() {
                    AstNode::Ident { name, .. } => {
                        self.emit_load(name, block, range);
                    }
                    _ => unreachable!(),
                }

                Ok(())
            }
            AstNode::IfExpr {
                cond,
                then,
                elifs,
                else_,
                range,
            } => {
                self.compile(cond, block)?;

                // Jump to the next condition if the first one doesn't match
                self.emit(block, Opcode::JumpIfFalse.into(), range);
                self.emit(block, 0xff, range);
                self.emit(block, 0xff, range);
                self.emit(block, 0xff, range);

                let then_jmp = block.size() - 3;

                self.compile(then, block)?;

                // Jump to the end of the if block once the then block has been executed
                self.emit(block, Opcode::Jump.into(), range);
                self.emit(block, 0xff, range);
                self.emit(block, 0xff, range);
                self.emit(block, 0xff, range);

                let mut end_jumps = Vec::new();

                let end_jmp = block.size() - 3;
                end_jumps.push(end_jmp);

                for (cond, body) in elifs.iter() {
                    self.patch_jump(then_jmp, block);

                    self.compile(cond, block)?;

                    // Jump to the next condition if the previous one doesn't match
                    self.emit(block, Opcode::JumpIfFalse.into(), range);
                    self.emit(block, 0xff, range);
                    self.emit(block, 0xff, range);
                    self.emit(block, 0xff, range);

                    let no_match_jmp = block.size() - 3;

                    self.compile(body, block)?;

                    // Jump to the end of the if block once the body has been executed
                    self.emit(block, Opcode::Jump.into(), range);
                    self.emit(block, 0xff, range);
                    self.emit(block, 0xff, range);
                    self.emit(block, 0xff, range);

                    let end_jmp = block.size() - 3;
                    end_jumps.push(end_jmp);

                    self.patch_jump(no_match_jmp, block);
                }

                if elifs.is_empty() {
                    self.patch_jump(then_jmp, block);
                }

                self.compile(else_, block)?;

                for end_jmp in end_jumps.iter() {
                    self.patch_jump(*end_jmp, block);
                }

                Ok(())
            }
            AstNode::CaseExpr { expr, cases, range } => {
                let mut end_jmps = Vec::new();

                for case in cases.iter() {
                    if let AstNode::CaseBranch { pattern, body, .. } = case {
                        self.compile(expr, block)?;

                        if let AstNode::CasePattern { kind, expr, .. } = pattern.as_ref() {
                            match kind {
                                CasePatternKind::Wildcard => {
                                    // Wildcard pattern always matches
                                    self.emit(block, Opcode::LoadTrue.into(), range);
                                }
                                CasePatternKind::Ident => {
                                    // Same as any, but also binds the value to a local
                                    if let AstNode::Ident { name, .. } = expr.as_ref() {
                                        self.emit_store(name.clone(), block, range, true);

                                        self.emit(block, Opcode::LoadTrue.into(), range);
                                    }
                                }
                                CasePatternKind::Literal => {
                                    self.compile(expr, block)?;
                                    self.emit(block, Opcode::Equals.into(), range);
                                }
                            }
                        }

                        // Jump to the next case if the pattern doesn't match
                        self.emit(block, Opcode::JumpIfFalse.into(), range);
                        self.emit(block, 0xff, range);
                        self.emit(block, 0xff, range);
                        self.emit(block, 0xff, range);

                        let no_match_jmp = block.size() - 3;

                        // If the pattern matches, execute the body
                        self.states.last_mut().unwrap().scope_depth += 1;
                        self.compile(body, block)?;
                        self.states.last_mut().unwrap().scope_depth -= 1;

                        // Leave the case block once the body has been executed
                        self.emit(block, Opcode::Jump.into(), range);
                        self.emit(block, 0xff, range);
                        self.emit(block, 0xff, range);
                        self.emit(block, 0xff, range);

                        let end_jmp = block.size() - 3;
                        end_jmps.push(end_jmp);

                        self.patch_jump(no_match_jmp, block);
                    }
                }

                // If no case matches, returns false
                self.emit(block, Opcode::LoadFalse.into(), range);

                for end_jmp in end_jmps.iter() {
                    self.patch_jump(*end_jmp, block);
                }

                Ok(())
            }
            AstNode::FuncCall {
                callee,
                args,
                range,
            } => {
                for arg in args.iter() {
                    self.compile(arg, block)?;
                }

                match callee.as_ref() {
                    AstNode::Ident { name, .. } => {
                        let op = match name.clone().as_str() {
                            "eq" => Some(Opcode::Equals),
                            "ne" => Some(Opcode::NotEquals),
                            "lt" => Some(Opcode::LessThan),
                            "lte" => Some(Opcode::LessThanOrEqual),
                            "gt" => Some(Opcode::GreaterThan),
                            "gte" => Some(Opcode::GreaterThanOrEqual),
                            "add" => Some(Opcode::Add),
                            "sub" => Some(Opcode::Subtract),
                            "mul" => Some(Opcode::Multiply),
                            "div" => Some(Opcode::Divide),
                            _ => None,
                        };
                        if let Some(op) = op {
                            self.emit(block, op.into(), range);
                            return Ok(());
                        }

                        if let Some(id) = self
                            .states
                            .iter()
                            .rev()
                            .rfind(|s| s.current_name == *name)
                            .map(|s| s.current_id)
                        {
                            // Whenever a function is called from within itself, we need to point
                            // the call to the correct closure, and since this might happen from
                            // within nested scopes, we iterate through every state to see if we
                            // get a match.
                            // I know, I know... This might not be the most efficient way to do
                            // this, but it works for now.
                            self.emit(block, Opcode::LoadConstant.into(), &range);
                            self.emit(block, id as u8, &range);

                            self.emit_call(args.len() as u8, block, range);
                        } else {
                            let is_final = self.emit_load(name, block, range);
                            if !is_final {
                                self.emit_call(args.len() as u8, block, range);
                            }
                        }
                    }
                    AstNode::FuncCall { .. } | AstNode::Lambda { .. } | AstNode::FuncRef { .. } => {
                        self.compile(callee, block)?;
                        self.emit_call(args.len() as u8, block, range);
                    }
                    _ => unreachable!(),
                }

                Ok(())
            }
            AstNode::IndexingOp { expr, index, range } => {
                self.compile(expr, block)?;
                self.compile(index, block)?;

                self.emit(block, Opcode::Indexing.into(), range);

                Ok(())
            }
            AstNode::ConcatOp { left, right, range } => {
                self.compile(left, block)?;
                self.compile(right, block)?;

                self.emit(block, Opcode::Concatenate.into(), range);

                Ok(())
            }
            AstNode::IterationOp {
                id,
                expr,
                body,
                range,
            } => {
                let id = match id.as_ref() {
                    AstNode::Ident { name, .. } => name.clone(),
                    _ => unreachable!(),
                };

                self.compile(expr, block)?;

                self.emit(block, Opcode::LoadIterator.into(), range);

                let loop_start = block.size() - 1;

                self.emit(block, Opcode::LoopIterator.into(), range);
                self.emit(block, 0xff, range);
                self.emit(block, 0xff, range);
                self.emit(block, 0xff, range);

                let end_jmp = block.size() - 3;

                let state = self.states.last_mut().unwrap();
                state.scope_depth += 1;

                let iter_local = Local {
                    name: EcoString::from("<iterator>"),
                    is_capture: false,
                    is_final: true,
                };
                state.locals.push(iter_local);

                let id_local = Local {
                    name: id,
                    is_capture: false,
                    is_final: true,
                };
                state.locals.push(id_local);

                self.compile(body, block)?;

                self.states.last_mut().unwrap().scope_depth -= 1;

                let loop_offset = block.size() - loop_start + 3;

                self.emit(block, Opcode::IteratorAppend.into(), range);
                self.emit(block, loop_offset as u8, range);
                self.emit(block, (loop_offset >> 8) as u8, range);
                self.emit(block, (loop_offset >> 16) as u8, range);

                // Discard hanging locals, as well as the iterator
                while self.states.last_mut().unwrap().locals.len() > 2 {
                    self.emit(block, Opcode::Pop.into(), range);
                    self.states.last_mut().unwrap().locals.pop();
                }

                self.patch_jump(end_jmp, block);

                return Ok(());
            }
            AstNode::Ident { name, range } => {
                self.emit_load(name, block, range);

                Ok(())
            }
            AstNode::BoolLit { value, range } => {
                match value {
                    true => self.emit(block, Opcode::LoadTrue.into(), range),
                    false => self.emit(block, Opcode::LoadFalse.into(), range),
                }

                Ok(())
            }
            AstNode::StrLit { value, range } => {
                let value = Rc::new(Value::String(value.clone()));

                let id = self.add_constant(value);
                self.emit(block, Opcode::LoadConstant.into(), range);
                self.emit(block, id as u8, range);

                Ok(())
            }
            AstNode::FmtString { parts, range } => {
                for part in parts.iter().rev() {
                    match part {
                        AstNode::StrLit { value, .. } => {
                            let value = Rc::new(Value::String(value.clone()));

                            let id = self.add_constant(value);
                            // println!("const line: {:?}", {
                            //     let range = match self.original_range {
                            //         Some(range) => range,
                            //         None => *range,
                            //     };
                            //     self.line_positions.from_offset(range.0).as_usize() + 1
                            // });
                            self.emit(block, Opcode::LoadConstant.into(), range);
                            self.emit(block, id as u8, range);
                        }
                        _ => {
                            self.original_range = Some(range.clone());
                            // println!("part line: {:?}", {
                            //     let range = match self.original_range {
                            //         Some(range) => range,
                            //         None => *range,
                            //     };
                            //     self.line_positions.from_offset(range.0).as_usize() + 1
                            // });
                            self.compile(part, block)?;
                            self.original_range = None;
                        }
                    }
                }

                self.emit(block, Opcode::BuildString.into(), range);
                self.emit(block, parts.len() as u8, range);

                Ok(())
            }
            AstNode::NumLit { value, range } => {
                match *value as usize {
                    0 => self.emit(block, Opcode::LoadZero.into(), range),
                    1 => self.emit(block, Opcode::LoadOne.into(), range),
                    2 => self.emit(block, Opcode::LoadTwo.into(), range),
                    _ => {
                        let value = Rc::new(Value::Number(*value));

                        let id = self.add_constant(value);
                        self.emit(block, Opcode::LoadConstant.into(), range);
                        self.emit(block, id as u8, range);
                    }
                }

                Ok(())
            }
            AstNode::List { items, range } => {
                for item in items.iter().rev() {
                    self.compile(item, block)?;
                }

                let size = items.len();

                self.emit(block, Opcode::LoadList.into(), range);
                self.emit(block, size as u8, range);
                self.emit(block, (size >> 8) as u8, range);

                Ok(())
            }
            AstNode::Record {
                keys,
                values,
                range,
            } => {
                for (key, value) in keys.iter().zip(values.iter()).rev() {
                    self.compile(value, block)?;
                    self.compile(key, block)?;
                }

                let size = keys.len();

                self.emit(block, Opcode::LoadRecord.into(), range);
                self.emit(block, size as u8, range);
                self.emit(block, (size >> 8) as u8, range);

                Ok(())
            }
            AstNode::Range {
                mode,
                start,
                end,
                step,
                range,
            } => {
                self.compile(start, block)?;
                self.compile(end, block)?;

                if let Some(step) = step {
                    self.compile(step, block)?;
                } else {
                    self.emit(block, Opcode::LoadOne.into(), range);
                }

                match mode {
                    RangeMode::Inclusive => {
                        self.emit(block, Opcode::LoadRangeInclusive.into(), range)
                    }
                    RangeMode::Exclusive => {
                        self.emit(block, Opcode::LoadRangeExclusive.into(), range)
                    }
                }

                Ok(())
            }
            _ => todo!("compile {:#?}", node),
        }
    }

    pub fn constants(&self) -> &Vec<Rc<Value>> {
        &self.constants
    }

    fn emit(&mut self, block: &mut Block, op: u8, range: &CodeRange) {
        let range = match self.original_range {
            Some(range) => range,
            None => *range,
        };
        let line = self.line_positions.from_offset(range.0).as_usize() + 1;

        block.insert(op, line);
    }

    fn emit_load(&mut self, name: &EcoString, block: &mut Block, range: &CodeRange) -> bool {
        let range = match self.original_range {
            Some(range) => range,
            None => *range,
        };

        let state = self.states.last().unwrap();

        // Try to find the variable in the local scope
        if let Some((idx, is_final)) = state
            .locals
            .iter()
            .enumerate()
            .rfind(|(_, l)| l.name == *name)
            .map(|(idx, l)| (idx, l.is_final))
        {
            let captures_self = state.locals.iter().any(|l| l.name == state.current_name);
            let idx = if captures_self { idx - 1 } else { idx };

            self.emit(block, Opcode::LoadLocal.into(), &range);
            self.emit(block, idx as u8, &range);

            return is_final;
        }

        // Try to find an upvalue
        if let Some((idx, is_final)) = self.resolve_upvalue(name, block, self.states.len() - 1) {
            self.emit(block, Opcode::LoadUpvalue.into(), &range);
            self.emit(block, idx as u8, &range);

            return is_final;
        }

        // Otherwise, assume it's a global
        let name = Rc::new(Value::String(name.clone()));

        let name_id = self.add_constant(name);
        self.emit(block, Opcode::LoadConstant.into(), &range);
        self.emit(block, name_id as u8, &range);

        self.emit(block, Opcode::LoadGlobal.into(), &range);

        let is_final = self.final_globals.contains(&name_id);
        is_final
    }

    fn emit_store(
        &mut self,
        name: EcoString,
        block: &mut Block,
        range: &CodeRange,
        is_final: bool,
    ) {
        let range = match self.original_range {
            Some(range) => range,
            None => *range,
        };

        let state = &mut self.states.last_mut().unwrap();
        if state.scope_depth > 0 {
            if state.locals.iter().rposition(|l| l.name == name).is_some() {
                panic!("redeclaration of local variable {}", name);
            }

            let local = Local {
                name,
                is_capture: false,
                is_final,
            };
            state.locals.push(local);
        } else {
            let name_id = self.add_constant(Value::String(name).into());
            self.emit(block, Opcode::LoadConstant.into(), &range);
            self.emit(block, name_id as u8, &range);

            self.emit(block, Opcode::StoreGlobal.into(), &range);

            if is_final {
                self.final_globals.push(name_id);
            }
        }
    }

    fn emit_closure(
        &mut self,
        name: Option<&EcoString>,
        params: &EcoVec<AstNode>,
        body: &AstNode,
        block: &mut Block,
        range: &CodeRange,
    ) -> CResult {
        let range = match self.original_range {
            Some(range) => range,
            None => *range,
        };

        self.states.push(CompilerState::new());

        let name = if let Some(name) = name {
            name.clone()
        } else {
            // Anonymous functions get a unique name so they can be distinguished
            // and captured correctly
            let last_state = &mut self.states.iter_mut().nth_back(1).unwrap();
            last_state.lambda_id += 1;

            EcoString::from(format!("<λ {:08x}>", last_state.lambda_id - 1))
        };

        let self_id = EcoString::from(format!("<tmp {}>", name.clone()));
        let self_id = self.add_constant(Rc::new(Value::String(self_id))); // Placeholder

        // This is necessary for self-references
        self.states.last_mut().unwrap().current_name = name.clone();
        self.states.last_mut().unwrap().current_id = self_id;

        let mut closure_block = Block::new();

        let closure_depth = self.states.iter().nth_back(1).unwrap().scope_depth + 1;
        self.states.last_mut().unwrap().scope_depth = closure_depth;

        let closure_state = &mut self.states.last_mut().unwrap();
        for param in params.iter() {
            let name = match param {
                AstNode::Ident { name, .. } => name.clone(),
                _ => unreachable!(),
            };

            let local = Local {
                name,
                is_capture: false,
                is_final: true,
            };
            closure_state.locals.push(local);
        }

        self.compile(body, &mut closure_block)?;

        self.emit(&mut closure_block, Opcode::Return.into(), &range);

        self.optimize(&mut closure_block);

        if DEBUG {
            println!("[{}]\n{}", name, closure_block);

            // println!(
            //     "locals for \"{}\": {:#?}",
            //     name,
            //     self.states.last_mut().unwrap().locals
            // );
        }

        let upvalue_refs = &self.states.last().unwrap().upvalue_refs.clone();
        let closure = Closure {
            name: name.clone(),
            arity: params.len(),
            chunk: closure_block,
            upvalues: Vec::new(),
            upvalue_refs: upvalue_refs.clone(),
        };
        let closure = Rc::new(Value::Closure(Rc::new(RefCell::new(closure))));

        self.states.pop();

        // Store the resulting closure in the scope
        self.set_constant(self_id, closure);
        self.emit(block, Opcode::LoadConstant.into(), &range);
        self.emit(block, self_id as u8, &range);

        self.emit_store(name, block, &range, false);

        // If the function captures any upvalues, we need to turn it into a closure
        if !upvalue_refs.is_empty() {
            self.emit(block, Opcode::LoadClosure.into(), &range);
            self.emit(block, self_id as u8, &range);
        }

        Ok(())
    }

    fn emit_call(&mut self, argc: u8, block: &mut Block, range: &CodeRange) {
        let range = match self.original_range {
            Some(range) => range,
            None => *range,
        };

        self.emit(block, Opcode::Call.into(), &range);
        self.emit(block, argc, &range);
    }

    fn add_constant(&mut self, value: Rc<Value>) -> usize {
        if let Some(idx) = self.constants.iter().rposition(|v| v == &value) {
            return idx;
        }

        self.constants.push(value);

        let id = self.constants.len() - 1;
        id
    }

    fn set_constant(&mut self, idx: usize, value: Rc<Value>) {
        self.constants[idx] = value;
    }

    fn patch_jump(&mut self, from: usize, block: &mut Block) {
        let offset = block.size() - from - 3;
        block.set(from, offset as u8);
        block.set(from + 1, (offset >> 8) as u8);
        block.set(from + 2, (offset >> 16) as u8);
    }

    fn resolve_upvalue(
        &mut self,
        name: &EcoString,
        block: &mut Block,
        state_index: usize,
    ) -> Option<(usize, bool)> {
        if state_index == 0 {
            // Reaching the root level means we looking at a global
            return None;
        }

        let state = &mut self.states.get_mut(state_index - 1).unwrap();
        if let Some((idx, local)) = state
            .locals
            .iter_mut()
            .enumerate()
            .rfind(|(_, l)| l.name == *name)
        {
            if DEBUG {
                println!("upvalue (local): {}", name);
            }

            local.is_capture = true;

            let is_final = local.is_final;
            let idx = self.add_upvalue(idx as u8, true, state_index);
            return Some((idx, is_final));
        }

        if let Some((idx, is_final)) = self.resolve_upvalue(name, block, state_index - 1) {
            if DEBUG {
                println!("upvalue (upvalue): {}", name);
            }

            let idx = self.add_upvalue(idx as u8, false, state_index);
            return Some((idx, is_final));
        }

        None
    }

    fn add_upvalue(&mut self, index: u8, is_local: bool, state_index: usize) -> usize {
        let state = self.states.get_mut(state_index).unwrap();

        if let Some(idx) = state
            .upvalue_refs
            .iter()
            .rposition(|u| u.index == index && u.is_local == is_local)
        {
            // Avoid duplicates
            return idx;
        }

        state.upvalue_refs.push(UpvalueRef { index, is_local });

        state.upvalue_refs.len() - 1
    }

    fn optimize(&self, block: &mut Block) {
        // TODO:
        // - [ ] Consume `NOP` instructions, adjusting jump offsets to compensate

        if DEBUG {
            // Skip optimization in debug mode
            return;
        }

        for i in 0..block.size() {
            let op = block.code()[i];
            if op >= Opcode::LAST as u8 {
                continue;
            }

            let op = Opcode::from(op);
            match op {
                Opcode::Jump => {
                    let offset = (block.code()[i + 1] as usize)
                        | ((block.code()[i + 2] as usize) << 8)
                        | ((block.code()[i + 3] as usize) << 16);

                    let op_at_offset = Opcode::from(block.code()[i + offset + 4]);
                    match op_at_offset {
                        Opcode::Return => {
                            // A jump to a return statement can be replaced with a return
                            block.set(i, Opcode::Return.into());
                            block.set(i + 1, Opcode::Nop.into());
                            block.set(i + 2, Opcode::Nop.into());
                            block.set(i + 3, Opcode::Nop.into());
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        }
    }
}
