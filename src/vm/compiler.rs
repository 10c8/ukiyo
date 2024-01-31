use std::{cell::RefCell, rc::Rc};

use ecow::{EcoString, EcoVec};

use crate::parser::{AstNode, CasePatternKind};

use super::{chunk::Chunk, Opcode, Upvalue, Value};

#[derive(Debug, Clone)]
pub struct Local {
    name: EcoString,
    #[allow(dead_code)]
    depth: usize,
    is_capture: bool,
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
    pub chunk: Chunk,
    pub upvalues: Vec<Rc<RefCell<Upvalue>>>,
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
}

impl CompilerState {
    pub fn new() -> Self {
        Self {
            locals: Vec::new(),
            scope_depth: 0,
            upvalue_refs: Vec::new(),
            lambda_id: 0,
        }
    }
}

pub struct Compiler {
    states: Vec<CompilerState>,
}

impl Compiler {
    pub fn new() -> Self {
        Self {
            states: vec![CompilerState::new()],
        }
    }

    pub fn compile(&mut self, node: &AstNode, chunk: &mut Chunk) -> CResult {
        match node {
            AstNode::Prog { body, .. } => {
                for (i, node) in body.iter().enumerate() {
                    self.compile(node, chunk)?;

                    if i < body.len() - 1 {
                        match node {
                            AstNode::ExprStmt { .. } => {
                                // Discard the result of every expression statement except
                                // the last one, which is the return value of the program.
                                // This is done so the stack doesn't get cluttered with unused
                                // values.
                                chunk.write(Opcode::Pop.into(), 1);
                            }
                            _ => {}
                        }
                    }
                }

                chunk.write(Opcode::Return.into(), 1);

                Ok(())
            }
            AstNode::Block { body, .. } => {
                self.states.last_mut().unwrap().scope_depth += 1;

                for node in body.iter() {
                    self.compile(node, chunk)?;
                }

                self.states.last_mut().unwrap().scope_depth -= 1;

                // TODO: Handle locals in non-closure blocks
                // while !state.locals.is_empty()
                //     && state.locals.last().unwrap().depth > state.scope_depth
                // {
                //     chunk.write(Opcode::Pop.into(), 1);

                //     state.locals.pop();
                // }

                Ok(())
            }
            AstNode::ExprStmt { expr, .. } => self.compile(expr, chunk),
            AstNode::FuncDecl {
                id,
                is_const,
                params,
                body,
                ..
            } => {
                let name = match id.as_ref() {
                    AstNode::Ident { name, .. } => name.clone(),
                    _ => unreachable!(),
                };

                if *is_const {
                    // Constant functions run once at the time of declaration, so that
                    // later calls will always point to the same value
                    self.compile(body, chunk)?;
                    self.emit_store(name, chunk);
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

                    self.emit_closure(Some(&name), params, body, chunk)?;
                }

                Ok(())
            }
            AstNode::Lambda { params, body, .. } => self.emit_closure(None, params, body, chunk),
            AstNode::FuncRef { id, .. } => {
                match id.as_ref() {
                    AstNode::Ident { name, .. } => {
                        self.emit_load(name, chunk);
                    }
                    _ => unreachable!(),
                }

                Ok(())
            }
            AstNode::Case { expr, cases, .. } => {
                let mut end_jmps = Vec::new();

                for (i, case) in cases.iter().enumerate() {
                    if let AstNode::CaseBranch { pattern, body, .. } = case {
                        if i > 0 {
                            // Pop the result of the previous test
                            chunk.write(Opcode::Pop.into(), 1);
                        }

                        self.compile(expr, chunk)?;

                        if let AstNode::CasePattern { kind, expr, .. } = pattern.as_ref() {
                            match kind {
                                CasePatternKind::Wildcard => {
                                    // Wildcard pattern always matches
                                    chunk.write(Opcode::LoadTrue.into(), 1);
                                }
                                CasePatternKind::Ident => {
                                    // Same as any, but also binds the value to a local
                                    if let AstNode::Ident { name, .. } = expr.as_ref() {
                                        self.emit_store(name.clone(), chunk);

                                        chunk.write(Opcode::LoadTrue.into(), 1);
                                    }
                                }
                                CasePatternKind::Literal => {
                                    self.compile(expr, chunk)?;
                                    chunk.write(Opcode::Equals.into(), 1);
                                }
                            }
                        }

                        // Jump to the next case if the pattern doesn't match
                        chunk.write(Opcode::JumpIfFalse.into(), 1);
                        chunk.write(0xff, 1);
                        chunk.write(0xff, 1);
                        chunk.write(0xff, 1);

                        let no_match_jmp = chunk.code.len() - 3;

                        // If the pattern matches, pop the result of the test and execute the body
                        chunk.write(Opcode::Pop.into(), 1);

                        self.states.last_mut().unwrap().scope_depth += 1;

                        self.compile(body, chunk)?;

                        self.states.last_mut().unwrap().scope_depth -= 1;

                        // Leave the case block once the body has been executed
                        chunk.write(Opcode::Jump.into(), 1);
                        chunk.write(0xff, 1);
                        chunk.write(0xff, 1);
                        chunk.write(0xff, 1);

                        let end_jmp = chunk.code.len() - 3;
                        end_jmps.push(end_jmp);

                        self.patch_jump(no_match_jmp, chunk);
                    }
                }

                // If no case matches, returns false
                chunk.write(Opcode::LoadFalse.into(), 1);

                for end_jmp in end_jmps.iter() {
                    self.patch_jump(*end_jmp, chunk);
                }

                Ok(())
            }
            AstNode::FuncCall { callee, args, .. } => {
                for arg in args.iter() {
                    self.compile(arg, chunk)?;
                }

                match callee.as_ref() {
                    AstNode::Ident { name, .. } => {
                        self.emit_load(name, chunk);
                    }
                    AstNode::FuncCall { .. } | AstNode::Lambda { .. } | AstNode::FuncRef { .. } => {
                        self.compile(callee, chunk)?;
                    }
                    _ => unreachable!(),
                }

                chunk.write(Opcode::Call.into(), 1);
                chunk.write(args.len() as u8, 1);

                Ok(())
            }
            AstNode::ConcatOp { left, right, .. } => {
                self.compile(left, chunk)?;
                self.compile(right, chunk)?;

                chunk.write(Opcode::Concatenate.into(), 1);

                Ok(())
            }
            AstNode::Ident { name, .. } => {
                self.emit_load(name, chunk);

                Ok(())
            }
            AstNode::BoolLit { value, .. } => {
                match value {
                    true => chunk.write(Opcode::LoadTrue.into(), 1),
                    false => chunk.write(Opcode::LoadFalse.into(), 1),
                }

                Ok(())
            }
            AstNode::StrLit { value, .. } => {
                let value = Rc::new(Value::String(value.clone()));

                let id = chunk.add_constant(value);
                chunk.write(Opcode::LoadConstant.into(), 1);
                chunk.write(id as u8, 1);

                Ok(())
            }
            AstNode::NumLit { value, .. } => {
                match *value as usize {
                    0 => chunk.write(Opcode::LoadZero.into(), 1),
                    1 => chunk.write(Opcode::LoadOne.into(), 1),
                    2 => chunk.write(Opcode::LoadTwo.into(), 1),
                    _ => {
                        let value = Rc::new(Value::Number(*value));

                        let id = chunk.add_constant(value);
                        chunk.write(Opcode::LoadConstant.into(), 1);
                        chunk.write(id as u8, 1);
                    }
                }

                Ok(())
            }
            _ => todo!("compile {:?}", node),
        }
    }

    fn emit_load(&mut self, name: &EcoString, chunk: &mut Chunk) {
        // Try to find the variable in the local scope
        let state = self.states.last().unwrap();
        if let Some(idx) = state.locals.iter().rposition(|l| l.name == *name) {
            chunk.write(Opcode::LoadLocal.into(), 1);
            chunk.write(idx as u8, 1);

            return;
        }

        // Try to find an upvalue
        if let Some(idx) = self.resolve_upvalue(name, chunk, self.states.len() - 1) {
            chunk.write(Opcode::LoadUpvalue.into(), 1);
            chunk.write(idx as u8, 1);

            return;
        }

        // Otherwise, assume it's a global
        let name = Rc::new(Value::String(name.clone()));

        let name_id = chunk.add_constant(name);
        chunk.write(Opcode::LoadConstant.into(), 1);
        chunk.write(name_id as u8, 1);

        chunk.write(Opcode::LoadGlobal.into(), 1);
    }

    fn emit_store(&mut self, name: EcoString, chunk: &mut Chunk) {
        let state = &mut self.states.last_mut().unwrap();
        if state.scope_depth > 0 {
            if state.locals.iter().rposition(|l| l.name == name).is_some() {
                panic!("redeclaration of local variable {}", name);
            }

            let local = Local {
                name,
                depth: state.scope_depth,
                is_capture: false,
            };
            state.locals.push(local);
        } else {
            let name_id = chunk.add_constant(Value::String(name).into());
            chunk.write(Opcode::LoadConstant.into(), 1);
            chunk.write(name_id as u8, 1);

            chunk.write(Opcode::StoreGlobal.into(), 1);
        }
    }

    fn patch_jump(&mut self, from: usize, chunk: &mut Chunk) {
        let offset = chunk.code.len() - from - 3;
        chunk.code[from] = offset as u8;
        chunk.code[from + 1] = (offset >> 8) as u8;
        chunk.code[from + 2] = (offset >> 16) as u8;
    }

    fn emit_closure(
        &mut self,
        name: Option<&EcoString>,
        params: &EcoVec<AstNode>,
        body: &AstNode,
        chunk: &mut Chunk,
    ) -> CResult {
        self.states.push(CompilerState::new());

        let mut closure_chunk = Chunk::new();

        let closure_depth = self.states.iter().nth_back(1).unwrap().scope_depth + 1;
        self.states.last_mut().unwrap().scope_depth = closure_depth;

        for param in params.iter() {
            let name = match param {
                AstNode::Ident { name, .. } => name.clone(),
                _ => unreachable!(),
            };

            let closure_state = &mut self.states.last_mut().unwrap();

            let local = Local {
                name,
                depth: closure_depth,
                is_capture: false,
            };
            closure_state.locals.push(local);
        }

        self.compile(body, &mut closure_chunk)?;

        closure_chunk.write(Opcode::Return.into(), 1);

        let name = if let Some(name) = name {
            name.clone()
        } else {
            // Anonymous functions get a unique name so they can be distinguished
            // and captured correctly
            let last_state = &mut self.states.iter_mut().nth_back(1).unwrap();
            last_state.lambda_id += 1;

            EcoString::from(format!("<λ {:08x}>", last_state.lambda_id - 1))
        };

        let upvalue_refs = &self.states.last().unwrap().upvalue_refs.clone();
        let closure = Closure {
            name: name.clone(),
            arity: params.len(),
            chunk: closure_chunk,
            upvalues: Vec::new(),
            upvalue_refs: upvalue_refs.clone(),
        };
        let closure = Rc::new(Value::Closure(Rc::new(RefCell::new(closure))));

        self.states.pop();

        // Store the resulting closure in the scope
        let closure_id = chunk.add_constant(closure);
        chunk.write(Opcode::LoadConstant.into(), 1);
        chunk.write(closure_id as u8, 1);

        self.emit_store(name, chunk);

        // If the function captures any upvalues, we need to turn it into a closure
        if !upvalue_refs.is_empty() {
            chunk.write(Opcode::GenClosure.into(), 1);
            chunk.write(closure_id as u8, 1);

            for u in upvalue_refs.iter() {
                chunk.write(u.index, 1);
                chunk.write(u.is_local as u8, 1);
            }
        }

        Ok(())
    }

    fn resolve_upvalue(
        &mut self,
        name: &EcoString,
        chunk: &mut Chunk,
        state_index: usize,
    ) -> Option<usize> {
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
            println!("upvalue (local): {}", name);
            local.is_capture = true;

            let idx = self.add_upvalue(idx as u8, true, state_index);
            return Some(idx);
        }

        if let Some(idx) = self.resolve_upvalue(name, chunk, state_index - 1) {
            println!("upvalue (upvalue): {}", name);

            let idx = self.add_upvalue(idx as u8, false, state_index);
            return Some(idx);
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

        println!("upvalues: {:?}", state.upvalue_refs);

        state.upvalue_refs.len() - 1
    }
}
