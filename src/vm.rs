pub mod chunk;
pub mod compiler;

use std::{
    cell::RefCell,
    fmt::{Debug, Display, Formatter},
    rc::Rc,
};

use chunk::Chunk;
use ecow::EcoString;
use once_cell::unsync::Lazy;
use rustc_hash::FxHashMap;

use compiler::Closure;

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Opcode {
    LoadConstant,
    #[allow(dead_code)]
    LoadConstantLong,
    StoreGlobal,
    LoadLocal,
    LoadGlobal,
    LoadUpvalue,
    LoadTrue,
    LoadFalse,
    LoadZero,
    LoadOne,
    LoadTwo,
    #[allow(dead_code)]
    LoadMinusOne,
    Call,
    GenClosure,
    Pop,
    Jump,
    JumpIfFalse,
    Equals,
    #[allow(dead_code)]
    Negate,
    #[allow(dead_code)]
    Add,
    #[allow(dead_code)]
    Subtract,
    #[allow(dead_code)]
    Multiply,
    #[allow(dead_code)]
    Divide,
    Concatenate,
    Return,
    LAST,
}

impl From<u8> for Opcode {
    fn from(byte: u8) -> Self {
        assert!(byte <= Opcode::LAST as u8);
        unsafe { std::mem::transmute(byte) }
    }
}

impl Into<u8> for Opcode {
    fn into(self) -> u8 {
        self as u8
    }
}

#[derive(Clone, PartialEq)]
pub enum Value {
    Nil,
    Bool(bool),
    String(EcoString),
    Number(f64),
    Closure(Rc<RefCell<Closure>>),
}

impl Debug for Value {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Nil => write!(f, "nil"),
            Value::Bool(b) => write!(f, "{}", b),
            Value::String(s) => write!(f, "\"{}\"", s),
            Value::Number(n) => write!(f, "{:?}", n),
            Value::Closure(c) => {
                let c = c.borrow();
                match &c.name.chars().nth(0).unwrap() {
                    '<' => write!(f, "{}", c.name),
                    _ => write!(f, "<fn {}>", c.name),
                }
            }
        }
    }
}

impl Display for Value {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Nil => write!(f, "nil"),
            Value::Bool(b) => write!(f, "{}", b),
            Value::String(s) => write!(f, "{}", s),
            Value::Number(n) => write!(f, "{:?}", n),
            Value::Closure(c) => {
                let c = c.borrow();
                match &c.name.chars().nth(0).unwrap() {
                    '<' => write!(f, "{}", c.name),
                    _ => write!(f, "<fn {}>", c.name),
                }
            }
        }
    }
}

pub struct CallFrame {
    closure: Rc<RefCell<Closure>>,
    pc: usize,
    offset: usize,
}

impl CallFrame {
    pub fn get_closure(&self) -> &Rc<RefCell<Closure>> {
        &self.closure
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum UpvalueKind {
    Open,
    Closed,
}

#[derive(Debug, Clone)]
pub struct Upvalue {
    pub kind: UpvalueKind,
    pub slot: Option<usize>,
    pub value: Rc<Value>,
}

#[derive(Debug)]
pub enum VMError {
    RuntimeError,
}

type VMResult = Result<(), VMError>;

const NIL_VALUE: Lazy<Rc<Value>> = Lazy::new(|| Rc::new(Value::Nil));
const TRUE_VALUE: Lazy<Rc<Value>> = Lazy::new(|| Rc::new(Value::Bool(true)));
const FALSE_VALUE: Lazy<Rc<Value>> = Lazy::new(|| Rc::new(Value::Bool(false)));
const ZERO_VALUE: Lazy<Rc<Value>> = Lazy::new(|| Rc::new(Value::Number(0.0)));
const ONE_VALUE: Lazy<Rc<Value>> = Lazy::new(|| Rc::new(Value::Number(1.0)));
const TWO_VALUE: Lazy<Rc<Value>> = Lazy::new(|| Rc::new(Value::Number(2.0)));
const MINUS_ONE_VALUE: Lazy<Rc<Value>> = Lazy::new(|| Rc::new(Value::Number(-1.0)));

pub struct VM {
    frames: Vec<CallFrame>,
    stack: Vec<Rc<Value>>,
    globals: FxHashMap<EcoString, Rc<Value>>,
    open_upvalues: Vec<Rc<RefCell<Upvalue>>>,
}

impl VM {
    pub fn new() -> VM {
        VM {
            frames: Vec::new(),
            stack: Vec::new(),
            globals: FxHashMap::default(),
            open_upvalues: Vec::new(),
        }
    }

    pub fn interpret(&mut self, chunk: Chunk) -> VMResult {
        let frame = CallFrame {
            closure: Rc::new(RefCell::new(Closure {
                name: "<program>".into(),
                arity: 0,
                chunk,
                upvalues: Vec::new(),
                upvalue_refs: Vec::new(),
            })),
            pc: 0,
            offset: 0,
        };
        self.frames.push(frame);

        self.run()
    }

    fn run(&mut self) -> VMResult {
        loop {
            if let Some(op) = self.read_byte() {
                let op = Opcode::from(op);

                {
                    let frame = self.frames.last().unwrap();
                    let closure = frame.get_closure().borrow();
                    let chunk = &closure.chunk;
                    let pc = frame.pc;

                    let mut data = EcoString::from(format!(".. .. .. {:02x}", op as u8));

                    let instruction = match op {
                        Opcode::LoadConstant => {
                            if let Some(idx) = chunk.read(pc) {
                                if let Some(value) = chunk.get_constant(*idx as usize) {
                                    data = format!(".. .. {:02x} {:02x}", op as u8, idx).into();

                                    format!("LOAD_CONST {:06x} = {:?}", idx, value)
                                } else {
                                    panic!("unknown constant");
                                }
                            } else {
                                panic!("missing constant id");
                            }
                        }
                        Opcode::LoadConstantLong => {
                            let idx_lo = chunk.read(pc);
                            let idx_md = chunk.read(pc + 1);
                            let idx_hi = chunk.read(pc + 2);

                            let idx = if let (Some(idx_lo), Some(idx_md), Some(idx_hi)) =
                                (idx_lo, idx_md, idx_hi)
                            {
                                data = format!(
                                    "{:02x} {:02x} {:02x} {:02x}",
                                    op as u8, idx_lo, idx_md, idx_hi
                                )
                                .into();

                                (*idx_lo as usize)
                                    | ((*idx_md as usize) << 8)
                                    | ((*idx_hi as usize) << 16)
                            } else {
                                panic!("missing constant id");
                            };

                            if let Some(value) = chunk.get_constant(idx) {
                                format!("LOAD_CONST {:06x} = {:?}", idx, value)
                            } else {
                                panic!("unknown constant: {}", idx);
                            }
                        }
                        Opcode::StoreGlobal => "ST_GLOBAL".to_string(),
                        Opcode::LoadLocal => {
                            if let Some(idx) = chunk.read(pc) {
                                data = format!(".. .. {:02x} {:02x}", op as u8, idx).into();

                                let value = self.stack.get(frame.offset + *idx as usize).unwrap();
                                format!("LD_LOCAL {:02x} = {:?}", idx, value)
                            } else {
                                panic!("missing local id");
                            }
                        }
                        Opcode::LoadGlobal => "LD_GLOBAL".to_string(),
                        Opcode::LoadUpvalue => {
                            if let Some(idx) = chunk.read(pc) {
                                data = format!(".. .. {:02x} {:02x}", op as u8, idx).into();

                                let upvalue = closure.upvalues.get(*idx as usize).unwrap().borrow();
                                match upvalue.kind {
                                    UpvalueKind::Open => {
                                        let slot = upvalue.slot.unwrap();
                                        let value = self.stack.get(slot).unwrap();
                                        format!("LD_UPVAL {:02x} (OPEN) = {:?}", idx, value)
                                    }
                                    UpvalueKind::Closed => {
                                        let value = upvalue.value.clone();
                                        format!("LD_UPVAL {:02x} (CLOSED) = {:?}", idx, value)
                                    }
                                }
                            } else {
                                panic!("missing upvalue id");
                            }
                        }
                        Opcode::LoadZero => "LOAD_ZERO".to_string(),
                        Opcode::LoadOne => "LOAD_ONE".to_string(),
                        Opcode::LoadTwo => "LOAD_TWO".to_string(),
                        Opcode::LoadMinusOne => "LOAD_MINUS_ONE".to_string(),
                        Opcode::LoadTrue => "LOAD_TRUE".to_string(),
                        Opcode::LoadFalse => "LOAD_FALSE".to_string(),
                        Opcode::Call => {
                            if let Some(argc) = chunk.read(pc) {
                                data = format!(".. .. {:02x} {:02x}", op as u8, argc).into();

                                format!("CALL {:02x}", argc)
                            } else {
                                panic!("missing argument count");
                            }
                        }
                        Opcode::GenClosure => {
                            let idx = if let Some(idx) = chunk.read(pc) {
                                *idx as usize
                            } else {
                                panic!("missing function id");
                            };

                            let value = if let Some(value) = chunk.get_constant(idx) {
                                value.clone()
                            } else {
                                panic!("unknown function");
                            };

                            if let Value::Closure(closure) = value.as_ref() {
                                let closure = closure.borrow();

                                data = format!(".. .. .. {:02x}", op as u8).into();

                                let mut line = String::from(format!(
                                    "GEN_CLOSURE {} * {:03x}",
                                    value,
                                    closure.upvalue_refs.len()
                                ));

                                for upvalue in &closure.upvalue_refs {
                                    line.push_str(&format!(
                                        "\n                     |  {} {:03x}",
                                        if upvalue.is_local { "LOCAL" } else { "UPVAL" },
                                        upvalue.index
                                    ));
                                }

                                line
                            } else {
                                panic!("invalid function");
                            }
                        }
                        Opcode::Pop => "POP".to_string(),
                        Opcode::Jump => {
                            let offset_lo = chunk.read(pc);
                            let offset_md = chunk.read(pc + 1);
                            let offset_hi = chunk.read(pc + 2);

                            let offset =
                                if let (Some(offset_lo), Some(offset_mi), Some(offset_hi)) =
                                    (offset_lo, offset_md, offset_hi)
                                {
                                    data = format!(
                                        "{:02x} {:02x} {:02x} {:02x}",
                                        op as u8, offset_lo, offset_mi, offset_hi
                                    )
                                    .into();

                                    (*offset_lo as usize)
                                        | ((*offset_mi as usize) << 8)
                                        | ((*offset_hi as usize) << 16)
                                } else {
                                    panic!("missing jump offset");
                                };

                            format!("JMP {:06x}", offset)
                        }
                        Opcode::JumpIfFalse => {
                            let offset_lo = chunk.read(pc);
                            let offset_md = chunk.read(pc + 1);
                            let offset_hi = chunk.read(pc + 2);

                            let offset =
                                if let (Some(offset_lo), Some(offset_mi), Some(offset_hi)) =
                                    (offset_lo, offset_md, offset_hi)
                                {
                                    data = format!(
                                        "{:02x} {:02x} {:02x} {:02x}",
                                        op as u8, offset_lo, offset_mi, offset_hi
                                    )
                                    .into();

                                    (*offset_lo as usize)
                                        | ((*offset_mi as usize) << 8)
                                        | ((*offset_hi as usize) << 16)
                                } else {
                                    panic!("missing jump offset");
                                };

                            format!("JF {:06x}", offset)
                        }
                        Opcode::Equals => "EQ".to_string(),
                        Opcode::Negate => "NEG".to_string(),
                        Opcode::Add => "ADD".to_string(),
                        Opcode::Multiply => "MUL".to_string(),
                        Opcode::Divide => "DIV".to_string(),
                        Opcode::Subtract => "SUB".to_string(),
                        Opcode::Concatenate => "CONCAT".to_string(),
                        Opcode::Return => "RET".to_string(),
                        _ => todo!(),
                    };

                    let line = chunk.get_line(pc).unwrap();

                    println!("{:#04}  {:04} {}  {}", pc - 1, line, data, instruction);
                }

                match op {
                    Opcode::LoadConstant => self.op_ld_const(false)?,
                    Opcode::LoadConstantLong => self.op_ld_const(true)?,
                    Opcode::StoreGlobal => self.op_st_global()?,
                    Opcode::LoadLocal => self.op_ld_local()?,
                    Opcode::LoadGlobal => self.op_ld_global()?,
                    Opcode::LoadUpvalue => self.op_ld_upval()?,
                    Opcode::LoadTrue => self.stack.push(TRUE_VALUE.clone()),
                    Opcode::LoadFalse => self.stack.push(FALSE_VALUE.clone()),
                    Opcode::LoadZero => self.stack.push(ZERO_VALUE.clone()),
                    Opcode::LoadOne => self.stack.push(ONE_VALUE.clone()),
                    Opcode::LoadTwo => self.stack.push(TWO_VALUE.clone()),
                    Opcode::LoadMinusOne => self.stack.push(MINUS_ONE_VALUE.clone()),
                    Opcode::Call => self.op_call()?,
                    Opcode::GenClosure => self.op_gen_closure()?,
                    Opcode::Pop => self.op_pop()?,
                    Opcode::Jump => self.op_jmp()?,
                    Opcode::JumpIfFalse => self.op_jf()?,
                    Opcode::Equals => self.op_eq()?,
                    Opcode::Negate => self.op_neg()?,
                    Opcode::Add => self.op_add()?,
                    Opcode::Subtract => self.op_sub()?,
                    Opcode::Multiply => self.op_mul()?,
                    Opcode::Divide => self.op_div()?,
                    Opcode::Concatenate => self.op_concat()?,
                    Opcode::Return => {
                        let result = self.stack.pop().unwrap_or(NIL_VALUE.clone());

                        self.close_upvalues(self.frames.last().unwrap().offset);

                        let frame = self.frames.pop().unwrap();

                        if self.frames.is_empty() {
                            println!("\n>>> RESULT = {:?}", result);

                            println!(">>> GLOBALS");
                            for (name, value) in &self.globals {
                                println!("{} = {:?}", name, value);
                            }

                            println!(">>> STACK");
                            for value in &self.stack {
                                println!("{:?}", value);
                            }

                            return Ok(());
                        }

                        self.stack.truncate(frame.offset);
                        self.stack.push(result);
                    }
                    _ => todo!("unknown opcode: {:?}", op),
                }
            } else {
                break;
            }
        }

        Ok(())
    }

    fn read_byte(&mut self) -> Option<u8> {
        let frame = self.frames.last_mut().unwrap();

        let op = {
            let chunk = &frame.get_closure().borrow().chunk;

            if let Some(op) = chunk.read(frame.pc) {
                Some((*op).into())
            } else {
                None
            }
        };

        if op.is_some() {
            frame.pc += 1;
        }

        op
    }

    fn call_value(&mut self, callee: &Rc<Value>, argc: usize) {
        match callee.as_ref() {
            Value::Closure(value) => {
                let closure = value.borrow();

                if argc != closure.arity {
                    panic!(
                        "call_value: expected {} arguments, got {}",
                        closure.arity, argc
                    );
                }

                // println!("{}", closure.chunk);

                let frame = CallFrame {
                    closure: value.clone(),
                    pc: 0,
                    offset: (self.stack.len() as isize - argc as isize) as usize,
                };

                self.frames.push(frame);
            }
            _ => {
                if argc > 0 {
                    panic!("call_value: value is not callable ({})", callee);
                }

                self.stack.push(callee.clone());
            }
        }
    }

    fn close_upvalues(&mut self, last: usize) {
        let mut i = self.open_upvalues.len() as isize;
        loop {
            if i < 1 {
                break;
            }

            let mut u = self.open_upvalues.get(i as usize - 1).unwrap().borrow_mut();

            let slot = u.slot.unwrap();
            if slot < last {
                break;
            }

            (*u).kind = UpvalueKind::Closed;
            (*u).slot = None;
            (*u).value = self.stack.get(slot).unwrap().clone();

            i -= 1;
        }

        self.open_upvalues.truncate(i as usize);
    }

    // LD_CONST <id: u8>
    fn op_ld_const(&mut self, is_long: bool) -> VMResult {
        let id = if is_long {
            let id_lo = self.read_byte();
            let id_md = self.read_byte();
            let id_hi = self.read_byte();

            if let (Some(id_lo), Some(id_md), Some(id_hi)) = (id_lo, id_md, id_hi) {
                (id_lo as usize) | ((id_md as usize) << 8) | ((id_hi as usize) << 16)
            } else {
                println!("ld_const: missing constant id");
                return Err(VMError::RuntimeError);
            }
        } else {
            if let Some(id) = self.read_byte() {
                id as usize
            } else {
                println!("ld_const: missing constant id");
                return Err(VMError::RuntimeError);
            }
        };

        let frame = self.frames.last_mut().unwrap();
        let chunk = &frame.get_closure().borrow().chunk;

        if let Some(constant) = chunk.get_constant(id as usize) {
            self.stack.push(constant.clone());
            return Ok(());
        }

        println!("ld_const: unknown constant");
        Err(VMError::RuntimeError)
    }

    // ST_GLOBAL
    fn op_st_global(&mut self) -> VMResult {
        if let Some(name) = self.stack.pop() {
            match name.as_ref() {
                Value::String(name) => {
                    if let Some(value) = self.stack.pop() {
                        if self.globals.get(name).is_some() {
                            println!("st_global: redefinition of variable");
                            return Err(VMError::RuntimeError);
                        }

                        self.globals.insert(name.clone(), value.clone());
                        return Ok(());
                    } else {
                        println!("st_global: missing value");
                        return Err(VMError::RuntimeError);
                    }
                }
                _ => {
                    println!("st_global: variable name must be a string");
                    return Err(VMError::RuntimeError);
                }
            }
        }

        println!("st_global: invalid variable name");
        Err(VMError::RuntimeError)
    }

    // LD_LOCAL <idx: u8>
    fn op_ld_local(&mut self) -> VMResult {
        if let Some(idx) = self.read_byte() {
            let frame = self.frames.last().unwrap();

            if let Some(value) = self.stack.get(frame.offset + idx as usize) {
                // println!("id: {} (offset: {})", idx as usize, frame.offset);

                self.stack.push(value.clone());
                return Ok(());
            } else {
                println!("id: {} (offset: {})", idx as usize, frame.offset);

                println!("ld_local: undefined variable");
                return Err(VMError::RuntimeError);
            }
        }

        println!("ld_local: unknown variable");
        Err(VMError::RuntimeError)
    }

    // LD_GLOBAL
    fn op_ld_global(&mut self) -> VMResult {
        if let Some(name) = self.stack.pop() {
            match name.as_ref() {
                Value::String(name) => {
                    if let Some(value) = self.globals.get(name) {
                        self.stack.push(value.clone());
                        return Ok(());
                    } else {
                        println!("ld_global: undefined variable '{}'", name);
                        return Err(VMError::RuntimeError);
                    }
                }
                _ => unreachable!(),
            }
        }

        println!("ld_global invalid variable name");
        Err(VMError::RuntimeError)
    }

    // LD_UPVAL <idx: u8>
    fn op_ld_upval(&mut self) -> VMResult {
        if let Some(idx) = self.read_byte() {
            let frame = self.frames.last().unwrap();
            let closure = frame.get_closure().borrow();

            if let Some(upvalue) = closure.upvalues.get(idx as usize) {
                let upvalue = upvalue.borrow();
                let value = match upvalue.kind {
                    UpvalueKind::Open => {
                        let slot = upvalue.slot.unwrap();
                        self.stack.get(slot).unwrap().clone()
                    }
                    UpvalueKind::Closed => upvalue.value.clone(),
                };

                self.stack.push(value.clone());
                return Ok(());
            }

            println!("ld_upval: undefined variable");
            return Err(VMError::RuntimeError);
        }

        println!("ld_upval: invalid index");
        Err(VMError::RuntimeError)
    }

    // CALL <argc: u8>
    fn op_call(&mut self) -> VMResult {
        let argc = if let Some(argc) = self.read_byte() {
            argc as usize
        } else {
            println!("call: argument count expected");
            return Err(VMError::RuntimeError);
        };

        let callee = if let Some(callee) = self.stack.pop() {
            callee.clone()
        } else {
            println!("call: value expected");
            return Err(VMError::RuntimeError);
        };

        // println!(">>> CALL {}:{}", callee, argc);

        self.call_value(&callee, argc);

        return Ok(());
    }

    // GEN_CLOSURE
    fn op_gen_closure(&mut self) -> VMResult {
        let closure_id = if let Some(closure_id) = self.read_byte() {
            closure_id as usize
        } else {
            println!("gen_closure: missing constant id");
            return Err(VMError::RuntimeError);
        };

        let value = {
            let frame = self.frames.last().unwrap();
            let chunk = &frame.get_closure().borrow().chunk;

            if let Some(value) = chunk.get_constant(closure_id) {
                value.clone()
            } else {
                println!("gen_closure: unknown constant");
                return Err(VMError::RuntimeError);
            }
        };

        // println!(">>> GEN_CLOSURE {}", value);

        if let Value::Closure(closure) = value.as_ref() {
            let closure = &mut closure.borrow_mut();

            for _ in 0..closure.upvalue_refs.len() {
                let idx = if let Some(idx) = self.read_byte() {
                    idx
                } else {
                    println!("gen_closure: missing upval index");
                    return Err(VMError::RuntimeError);
                };

                let is_local = if let Some(is_local) = self.read_byte() {
                    is_local as u8 == 1
                } else {
                    println!("gen_closure: missing upval type");
                    return Err(VMError::RuntimeError);
                };

                let frame = self.frames.last().unwrap();
                let frame_closure = frame.get_closure().borrow();

                if is_local {
                    if self
                        .open_upvalues
                        .iter()
                        .rfind(|u| {
                            let u = u.borrow();
                            u.kind == UpvalueKind::Open
                                && u.slot.unwrap() == frame.offset + idx as usize
                        })
                        .is_none()
                    {
                        let value = if let Some(value) = self.stack.get(frame.offset + idx as usize)
                        {
                            value
                        } else {
                            println!("gen_closure: undefined variable");
                            println!("idx: {} (offset: {})", idx as usize, frame.offset);

                            return Err(VMError::RuntimeError);
                        };

                        let upvalue = Upvalue {
                            kind: UpvalueKind::Open,
                            slot: Some(frame.offset + idx as usize),
                            value: value.clone(),
                        };
                        let upvalue = Rc::new(RefCell::new(upvalue));

                        // println!(
                        //     "    capture LOCAL {:03} (offset: {}) = {}",
                        //     idx as u8, frame.offset, value
                        // );

                        closure.upvalues.push(upvalue.clone());
                        self.open_upvalues.push(upvalue);
                    }
                } else {
                    // println!(
                    //     "    ref UPVAL {} = {}",
                    //     idx as u8,
                    //     frame_closure
                    //         .upvalues
                    //         .get(idx as usize)
                    //         .unwrap()
                    //         .lock()
                    //         .unwrap()
                    //         .value,
                    // );

                    let u = frame_closure.upvalues.get(idx as usize).unwrap();
                    closure.upvalues.push(u.clone());
                }
            }

            self.stack.pop();
            self.stack.push(value.clone());

            return Ok(());
        } else {
            println!("gen_closure: invalid closure");
            Err(VMError::RuntimeError)
        }
    }

    // POP
    fn op_pop(&mut self) -> VMResult {
        if self.stack.pop().is_some() {
            return Ok(());
        }

        println!("pop: stack underflow");
        Err(VMError::RuntimeError)
    }

    // JMP <offset: u24>
    fn op_jmp(&mut self) -> VMResult {
        let offset_lo = self.read_byte();
        let offset_md = self.read_byte();
        let offset_hi = self.read_byte();

        let offset = if let (Some(offset_lo), Some(offset_md), Some(offset_hi)) =
            (offset_lo, offset_md, offset_hi)
        {
            (offset_lo as usize) | ((offset_md as usize) << 8) | ((offset_hi as usize) << 16)
        } else {
            println!("jmp: missing offset");
            return Err(VMError::RuntimeError);
        };

        let frame = self.frames.last_mut().unwrap();
        // println!("jmp: jump to {}", frame.pc + offset);
        frame.pc += offset;

        Ok(())
    }

    // JF <offset: u24>
    fn op_jf(&mut self) -> VMResult {
        let offset_lo = self.read_byte();
        let offset_md = self.read_byte();
        let offset_hi = self.read_byte();

        let offset = if let (Some(offset_lo), Some(offset_md), Some(offset_hi)) =
            (offset_lo, offset_md, offset_hi)
        {
            (offset_lo as usize) | ((offset_md as usize) << 8) | ((offset_hi as usize) << 16)
        } else {
            println!("jf: missing offset");
            return Err(VMError::RuntimeError);
        };

        if let Some(condition) = self.stack.get(self.stack.len() - 1) {
            if let Value::Bool(condition) = condition.as_ref() {
                if !condition {
                    let frame = self.frames.last_mut().unwrap();
                    // println!("jf: jump to {}", offset);
                    frame.pc += offset;
                }

                return Ok(());
            }
        }

        println!("jf: invalid condition");
        Err(VMError::RuntimeError)
    }

    // EQ
    fn op_eq(&mut self) -> VMResult {
        let b = self.stack.pop();
        let a = self.stack.pop();

        if let (Some(a), Some(b)) = (a, b) {
            self.stack.push(Value::Bool(a == b).into());
            return Ok(());
        }

        println!("eq: invalid operands");
        Err(VMError::RuntimeError)
    }

    // NEG
    fn op_neg(&mut self) -> VMResult {
        if let Some(value) = self.stack.pop() {
            if let Value::Number(n) = value.as_ref() {
                self.stack.push(Value::Number(-n).into());
                return Ok(());
            }
        }

        println!("neg: invalid operand");
        Err(VMError::RuntimeError)
    }

    // ADD
    fn op_add(&mut self) -> VMResult {
        let b = self.stack.pop();
        let a = self.stack.pop();

        if let (Some(a), Some(b)) = (a, b) {
            if let (Value::Number(a), Value::Number(b)) = (a.as_ref(), b.as_ref()) {
                self.stack.push(Value::Number(a + b).into());
                return Ok(());
            }
        }

        println!("add: invalid operands");
        Err(VMError::RuntimeError)
    }

    // SUB
    fn op_sub(&mut self) -> VMResult {
        let b = self.stack.pop();
        let a = self.stack.pop();

        if let (Some(a), Some(b)) = (a, b) {
            if let (Value::Number(a), Value::Number(b)) = (a.as_ref(), b.as_ref()) {
                self.stack.push(Value::Number(a - b).into());
                return Ok(());
            }
        }

        println!("sub: invalid operands");
        Err(VMError::RuntimeError)
    }

    // MUL
    fn op_mul(&mut self) -> VMResult {
        let b = self.stack.pop();
        let a = self.stack.pop();

        if let (Some(a), Some(b)) = (a, b) {
            if let (Value::Number(a), Value::Number(b)) = (a.as_ref(), b.as_ref()) {
                self.stack.push(Value::Number(a * b).into());
                return Ok(());
            }
        }

        println!("mul: invalid operands");
        Err(VMError::RuntimeError)
    }

    // DIV
    fn op_div(&mut self) -> VMResult {
        let b = self.stack.pop();
        let a = self.stack.pop();

        if let (Some(a), Some(b)) = (a, b) {
            if let (Value::Number(a), Value::Number(b)) = (a.as_ref(), b.as_ref()) {
                self.stack.push(Value::Number(a / b).into());
                return Ok(());
            }
        }

        println!("div: invalid operands");
        Err(VMError::RuntimeError)
    }

    // CONCAT
    fn op_concat(&mut self) -> VMResult {
        let b = self.stack.pop();
        let a = self.stack.pop();

        if let (Some(a), Some(b)) = (a, b) {
            let a = a.to_string();
            let b = b.to_string();

            let mut result = a.clone();
            result.push_str(&b);

            self.stack.push(Value::String(result.into()).into());

            return Ok(());
        }

        println!("concat: invalid operands");
        Err(VMError::RuntimeError)
    }
}
