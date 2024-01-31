use std::sync::Arc;

use ecow::{EcoString, EcoVec};

use crate::parser::AstNode;

use super::{chunk::Chunk, Closure, Opcode, Value};

#[derive(Debug, Clone)]
pub struct Local {
    name: EcoString,
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
    Unimplemented,
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
                for node in body.iter() {
                    self.compile(node, chunk)?;
                }

                chunk.write(Opcode::Return.into(), 1);

                Ok(())
            }
            AstNode::Block { body, .. } => {
                self.states.last_mut().unwrap().scope_depth += 1;

                for node in body.iter() {
                    self.compile(node, chunk)?;
                }

                let state = &mut self.states.last_mut().unwrap();

                state.scope_depth -= 1;

                println!("locals: {:#?}", state.locals);

                // TODO: Handle locals in non-closure blocks
                // while !state.locals.is_empty()
                //     && state.locals.last().unwrap().depth > state.scope_depth
                // {
                //     chunk.write(Opcode::Pop.into(), 1);

                //     state.locals.pop();
                // }

                Ok(())
            }
            AstNode::ExprStmt { expr, .. } => {
                self.compile(expr, chunk)?;

                Ok(())
            }
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
                    self.emit_store(name, chunk, self.states.last().unwrap().scope_depth);
                } else {
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
                    callee => todo!("compile callee {:?}", callee),
                }

                chunk.write(Opcode::Call.into(), 1);
                chunk.write(args.len() as u8, 1);

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
                let value = Arc::new(Value::String(value.clone()));

                let id = chunk.add_constant(value.clone());

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
                        let value = Arc::new(Value::Number(*value));

                        let id = chunk.add_constant(value.clone());

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
        let idx = chunk.add_constant(Value::String(name.clone()).into());

        chunk.write(Opcode::LoadConstant.into(), 1);
        chunk.write(idx as u8, 1);
        chunk.write(Opcode::LoadGlobal.into(), 1);
    }

    fn emit_store(&mut self, name: EcoString, chunk: &mut Chunk, depth: usize) {
        let state = &mut self.states.last_mut().unwrap();
        if depth > 0 {
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

    fn emit_closure(
        &mut self,
        name: Option<&EcoString>,
        params: &EcoVec<AstNode>,
        body: &AstNode,
        chunk: &mut Chunk,
    ) -> CResult {
        self.states.push(CompilerState::new());

        let mut closure_chunk = Chunk::new();
        let closure_depth = self.states.last().unwrap().scope_depth + 1;

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
        let closure = Arc::new(Value::Closure(closure.into()));

        self.states.pop();

        // Store the resulting closure in the scope
        let closure_id = chunk.add_constant(closure.clone());
        chunk.write(Opcode::LoadConstant.into(), 1);
        chunk.write(closure_id as u8, 1);

        self.emit_store(name, chunk, closure_depth);

        // If the function captures any upvalues, we need to turn it into a closure
        if !upvalue_refs.is_empty() {
            chunk.write(Opcode::Closure.into(), 1);
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
