pub mod block;
pub mod compiler;

#[allow(unused_imports)]
use std::{
    cell::RefCell,
    fmt::{write, Debug, Display, Formatter},
    rc::Rc,
};

use block::Block;
use ecow::EcoString;
use rand::Rng;
use rustc_hash::FxHashMap;

use compiler::Closure;

use crate::DEBUG;

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
    LoadClosure,
    LoadList,
    LoadRecord,
    LoadRangeInclusive,
    LoadRangeExclusive,
    Indexing,
    LoadIterator,
    LoopIterator,
    IteratorAppend,
    BuildString,
    Pop,
    #[allow(dead_code)]
    PopTop,
    Jump,
    JumpIfFalse,
    #[allow(dead_code)]
    JumpBack,
    Equals,
    NotEquals,
    LessThan,
    GreaterThan,
    LessThanOrEqual,
    GreaterThanOrEqual,
    #[allow(dead_code)]
    Negate,
    Add,
    Subtract,
    Multiply,
    Divide,
    Concatenate,
    Return,
    Nop,
    LAST,
}

impl From<u8> for Opcode {
    fn from(byte: u8) -> Self {
        assert!(byte <= Opcode::LAST as u8);

        // Safety: `byte` is guaranteed to be a valid `Opcode` variant.
        unsafe { std::mem::transmute(byte) }
    }
}

impl Into<u8> for Opcode {
    fn into(self) -> u8 {
        self as u8
    }
}

#[derive(Clone)]
pub struct IteratorInner {
    collection: Rc<Value>,
    index: usize,
    result: Vec<Rc<Value>>,
}

impl IteratorInner {
    pub fn new(collection: Rc<Value>) -> Self {
        Self {
            collection,
            index: 0,
            result: Vec::new(),
        }
    }

    pub fn next(&mut self) -> Option<Rc<Value>> {
        match Rc::get_mut(&mut self.collection).unwrap() {
            Value::List(collection) => {
                if self.index < collection.len() {
                    let value = collection[self.index].clone();
                    self.index += 1;

                    return Some(value);
                }
            }
            Value::RangeList(collection) => {
                if let Some(value) = collection.borrow_mut().get(self.index) {
                    self.index += 1;

                    return Some(value);
                }
            }
            _ => {}
        }

        None
    }

    pub fn push_result(&mut self, value: Rc<Value>) {
        self.result.push(value);
    }
}

impl PartialEq for IteratorInner {
    fn eq(&self, _other: &Self) -> bool {
        // Iterators are only used internally, so we don't need to compare them.
        false
    }
}

#[derive(Clone)]
pub struct RangeListInner {
    is_reverse: bool,
    values: Vec<Rc<Value>>,
    start: usize,
    end: usize,
    step: usize,
    size: usize,
    last: usize,
}

impl RangeListInner {
    pub fn new(start: usize, end: usize, step: usize) -> Self {
        let mut values = Vec::new();
        values.push(Rc::new(Value::Number(start as f64)));

        let size = if start < end {
            (end - start) / step
        } else {
            (start - end) / step
        } + 1;

        Self {
            is_reverse: start > end,
            values,
            start,
            end,
            step,
            size,
            last: 0,
        }
    }

    pub fn get(&mut self, index: usize) -> Option<Rc<Value>> {
        if index < self.size {
            if index > self.last {
                for i in self.last..index {
                    let n = if self.is_reverse {
                        self.start - ((i + 1) * self.step)
                    } else {
                        self.start + ((i + 1) * self.step)
                    };

                    self.values.push(Rc::new(Value::Number(n as f64)));
                }

                self.last = index;
            }

            return Some(self.values[index].clone());
        }

        None
    }

    pub fn len(&self) -> usize {
        self.size
    }

    pub fn resolve(&mut self) {
        if self.last < self.size {
            for i in self.last..self.size {
                let n = if self.is_reverse {
                    self.start - ((i + 1) * self.step)
                } else {
                    self.start + ((i + 1) * self.step)
                };

                self.values.push(Rc::new(Value::Number(n as f64)));
            }

            self.last = self.size;
        }

        self.last = self.size;
    }
}

impl Iterator for RangeListInner {
    type Item = Rc<Value>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.last < self.size {
            let n = self.start + self.last + 1;
            self.values.push(Rc::new(Value::Number(n as f64)));

            self.last += 1;

            return Some(self.values[self.last].clone());
        }

        None
    }
}

impl PartialEq for RangeListInner {
    fn eq(&self, other: &Self) -> bool {
        self.start == other.start && self.end == other.end
    }
}

impl Debug for RangeListInner {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}..={}]", self.start, self.end)
    }
}

#[derive(Clone, PartialEq)]
pub enum Value {
    Nil,
    Bool(bool),
    String(EcoString),
    Number(f64),
    List(Vec<Rc<Value>>),
    Record(FxHashMap<EcoString, Rc<Value>>),
    MutableRecord(Rc<RefCell<FxHashMap<EcoString, Rc<Value>>>>),
    RangeList(Rc<RefCell<RangeListInner>>),
    Iterator(Rc<RefCell<IteratorInner>>),
    Closure(Rc<RefCell<Closure>>),
    NativeFunction {
        function: fn(&'static mut VM),
        arity: usize,
    },
}

impl Value {
    #[allow(dead_code)]
    pub fn get_type(&self) -> &'static str {
        match self {
            Value::Nil => "nil",
            Value::Bool(_) => "bool",
            Value::String(_) => "string",
            Value::Number(_) => "number",
            Value::List(_) => "list",
            Value::Record(_) => "record",
            Value::MutableRecord(_) => "mutable record",
            Value::RangeList(_) => "range",
            Value::Iterator(_) => "iterator",
            Value::Closure(_) => "closure",
            Value::NativeFunction { .. } => "native",
        }
    }
}

impl Debug for Value {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Nil => write!(f, "nil"),
            Value::Bool(value) => write!(f, "{}", value),
            Value::String(value) => write!(f, "\"{}\"", value),
            Value::Number(value) => write!(f, "{:?}", value),
            Value::List(value) => write!(f, "{:?}", value),
            Value::Record(value) => write!(f, "{:?}", value),
            Value::MutableRecord(value) => write!(f, "{:?}", value.borrow()),
            Value::RangeList(value) => write!(f, "{:?}", value.borrow()),
            Value::Iterator(_) => write!(f, "<iterator>"),
            Value::Closure(value) => {
                let closure = value.try_borrow();
                let name = if let Ok(closure) = closure {
                    closure.name.clone()
                } else {
                    "<self>".into()
                };

                match name.chars().nth(0).unwrap() {
                    '<' => write!(f, "{}", name),
                    _ => write!(f, "<fn {}>", name),
                }
            }
            Value::NativeFunction { .. } => write!(f, "<native fn>"),
        }
    }
}

impl Display for Value {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Nil => write!(f, "nil"),
            Value::Bool(value) => write!(f, "{}", value),
            Value::String(value) => write!(f, "{}", value),
            Value::Number(value) => write!(f, "{:?}", value),
            Value::List(items) => {
                write!(f, "[")?;

                for (i, value) in items.iter().enumerate() {
                    write!(f, "{}", value)?;

                    if i < items.len() - 1 {
                        write!(f, ", ")?;
                    }
                }

                write!(f, "]")
            }
            Value::Record(entries) => {
                write!(f, "{{")?;

                for (i, (key, value)) in entries.iter().enumerate() {
                    write!(f, "{}: {}", key, value)?;

                    if i < entries.len() - 1 {
                        write!(f, ", ")?;
                    }
                }

                write!(f, "}}")
            }
            Value::MutableRecord(_) => write!(f, "<mut record>"),
            Value::RangeList(value) => write!(f, "{:?}", value.borrow()),
            Value::Iterator(_) => write!(f, "<iterator>"),
            Value::Closure(value) => {
                let closure = value.borrow();
                match &closure.name.chars().nth(0).unwrap() {
                    '<' => write!(f, "{}", closure.name),
                    _ => write!(f, "<fn {}>", closure.name),
                }
            }
            Value::NativeFunction { .. } => write!(f, "<native fn>"),
        }
    }
}

pub struct Frame {
    closure: Rc<RefCell<Closure>>,
    pc: usize,
    offset: usize,
}

impl Frame {
    pub fn get_closure(&self) -> &Rc<RefCell<Closure>> {
        &self.closure
    }
}

#[derive(Debug, Clone)]
pub struct Upvalue {
    pub slot: usize,
    pub value: Option<Rc<Value>>,
}

macro_rules! std_fn {
    ($self:ident: $name:ident, $arity:expr) => {
        $self.globals.insert(
            stringify!($name).chars().skip(4).collect(),
            Rc::new(Value::NativeFunction {
                function: VM::$name,
                arity: $arity,
            }),
        );
    };
}

#[derive(Debug)]
pub enum VMError {
    RuntimeError,
}

type VMResult = Result<(), VMError>;

pub struct VM {
    frames: Vec<Frame>,
    stack: Vec<Rc<Value>>,
    globals: FxHashMap<EcoString, Rc<Value>>,
    constants: Vec<Rc<Value>>,

    rand_thread: rand::rngs::ThreadRng,

    nil_value: Rc<Value>,
    true_value: Rc<Value>,
    false_value: Rc<Value>,
    zero_value: Rc<Value>,
    one_value: Rc<Value>,
    two_value: Rc<Value>,
    minus_one_value: Rc<Value>,
}

impl VM {
    pub fn new(constants: Vec<Rc<Value>>) -> Self {
        Self {
            frames: Vec::new(),
            stack: Vec::new(),
            globals: FxHashMap::default(),
            constants,

            rand_thread: rand::thread_rng(),

            nil_value: Rc::new(Value::Nil),
            true_value: Rc::new(Value::Bool(true)),
            false_value: Rc::new(Value::Bool(false)),
            zero_value: Rc::new(Value::Number(0.0)),
            one_value: Rc::new(Value::Number(1.0)),
            two_value: Rc::new(Value::Number(2.0)),
            minus_one_value: Rc::new(Value::Number(-1.0)),
        }
    }

    pub fn load_stdlib(&mut self) {
        std_fn!(self: std_trace, 1);

        std_fn!(self: std_str, 1);

        std_fn!(self: std_mrecord_new, 1);
        std_fn!(self: std_mrecord_read, 2);
        std_fn!(self: std_mrecord_write, 3);
        std_fn!(self: std_mrecord_has, 2);

        std_fn!(self: std_compare, 2);

        std_fn!(self: std_pow, 2);

        std_fn!(self: std_len, 1);
        std_fn!(self: std_head, 1);
        std_fn!(self: std_tail, 1);
        std_fn!(self: std_take, 2);
        std_fn!(self: std_drop, 2);

        std_fn!(self: std_rand, 2);
    }

    fn std_trace(&mut self) {
        let value = self.stack.pop().unwrap();

        println!("{}", value);

        self.stack.push(self.nil_value.clone());
    }

    fn std_str(&mut self) {
        let value = self.stack.pop().unwrap();
        let value = EcoString::from(value.to_string());

        let result = Rc::new(Value::String(value));
        self.stack.push(result);
    }

    fn std_mrecord_new(&mut self) {
        let initial = self.stack.pop().unwrap();
        let initial = if let Value::Record(entries) = initial.as_ref() {
            Rc::new(RefCell::new(entries.clone()))
        } else {
            panic!("mrecord_new: invalid record");
        };

        let result = Rc::new(Value::MutableRecord(initial));
        self.stack.push(result);
    }

    fn std_mrecord_read(&mut self) {
        let key = self.stack.pop().unwrap();
        let record = self.stack.pop().unwrap();

        let key = if let Value::String(value) = key.as_ref() {
            value.clone()
        } else {
            panic!("mrecord_read: invalid key");
        };

        if let Value::MutableRecord(record) = record.as_ref() {
            let record = record.borrow();

            if let Some(value) = record.get(&key) {
                self.stack.push(value.clone());
            } else {
                panic!(
                    "read_recmrecord_readord: key \"{}\" not found in record",
                    key
                );
            }
        } else {
            panic!("mrecord_read: mutable record expected");
        }
    }

    fn std_mrecord_write(&mut self) {
        let value = self.stack.pop().unwrap();
        let key = self.stack.pop().unwrap();
        let record = self.stack.pop().unwrap();

        let key = if let Value::String(value) = key.as_ref() {
            value.clone()
        } else {
            panic!("mrecord_write: invalid key");
        };

        if let Value::MutableRecord(record) = record.as_ref() {
            let mut record = record.borrow_mut();
            record.insert(key, value.clone());
        } else {
            panic!("mrecord_write: mutable record expected");
        }

        self.stack.push(value);
    }

    fn std_mrecord_has(&mut self) {
        let key = self.stack.pop().unwrap();
        let record = self.stack.pop().unwrap();

        let key = if let Value::String(value) = key.as_ref() {
            value.clone()
        } else {
            panic!("mrecord_has: invalid key");
        };

        if let Value::MutableRecord(record) = record.as_ref() {
            let record = record.borrow();

            if record.contains_key(&key) {
                self.stack.push(self.true_value.clone());
            } else {
                self.stack.push(self.false_value.clone());
            }
        } else {
            panic!("mrecord_read: mutable record expected");
        }
    }

    fn std_compare(&mut self) {
        let b = self.stack.pop().unwrap();
        let a = self.stack.pop().unwrap();

        if let Value::Number(a) = a.as_ref() {
            if let Value::Number(b) = b.as_ref() {
                let result = if a < b {
                    "lt"
                } else if a > b {
                    "gt"
                } else {
                    "eq"
                };

                self.stack.push(Rc::new(Value::String(result.into())));
                return;
            }
        }

        println!("a: {:?}, b: {:?}", a, b);

        panic!("compare: invalid argument");
    }

    fn std_pow(&mut self) {
        let exponent = self.stack.pop().unwrap();
        let base = self.stack.pop().unwrap();

        if let Value::Number(base) = base.as_ref() {
            if let Value::Number(exponent) = exponent.as_ref() {
                let result = base.powf(*exponent) as usize;
                self.stack.push(Rc::new(Value::Number(result as f64)));
                return;
            }
        }

        panic!("pow: invalid argument");
    }

    fn std_len(&mut self) {
        let list = self.stack.pop().unwrap();

        let result = match list.as_ref() {
            Value::List(list) => list.len(),
            Value::RangeList(range) => range.borrow().len(),
            _ => panic!("len: invalid argument"),
        };

        self.stack.push(Rc::new(Value::Number(result as f64)));
    }

    fn std_head(&mut self) {
        let list = self.stack.pop().unwrap();

        let result = match list.as_ref() {
            Value::List(list) => list.first().cloned(),
            Value::RangeList(range) => range.borrow_mut().get(0),
            _ => panic!("head: invalid argument"),
        };

        if let Some(result) = result {
            self.stack.push(result);
            return;
        }

        panic!("head: empty list");
    }

    fn std_tail(&mut self) {
        let list = self.stack.pop().unwrap();

        let result = match list.as_ref() {
            Value::List(list) => {
                let mut result = Vec::new();

                for i in 1..list.len() {
                    if let Some(value) = list.get(i) {
                        result.push(value.clone());
                    } else {
                        break;
                    }
                }

                Rc::new(Value::List(result))
            }
            Value::RangeList(range) => {
                let mut range = range.borrow_mut();

                let mut result = Vec::new();

                for i in 1..range.len() {
                    if let Some(value) = range.get(i) {
                        result.push(value.clone());
                    } else {
                        break;
                    }
                }

                Rc::new(Value::List(result))
            }
            _ => panic!("tail: invalid argument"),
        };

        self.stack.push(result);
    }

    fn std_take(&mut self) {
        let count = self.stack.pop().unwrap();
        let list = self.stack.pop().unwrap();

        if let Value::Number(count) = count.as_ref() {
            let count = *count as usize;

            let result = match list.as_ref() {
                Value::List(list) => {
                    let mut result = Vec::new();

                    for i in 0..count {
                        if let Some(value) = list.get(i) {
                            result.push(value.clone());
                        } else {
                            break;
                        }
                    }

                    Rc::new(Value::List(result))
                }
                Value::RangeList(range) => {
                    let mut result = Vec::new();

                    for i in 0..count {
                        if let Some(value) = range.borrow_mut().get(i) {
                            result.push(value);
                        } else {
                            break;
                        }
                    }

                    Rc::new(Value::List(result))
                }
                _ => panic!("take: invalid argument"),
            };

            self.stack.push(result);
            return;
        }

        panic!("take: invalid argument");
    }

    fn std_drop(&mut self) {
        let count = self.stack.pop().unwrap();
        let list = self.stack.pop().unwrap();

        if let Value::Number(count) = count.as_ref() {
            let count = *count as usize;

            let result = match list.as_ref() {
                Value::List(list) => {
                    let mut result = Vec::new();

                    for i in count..list.len() {
                        if let Some(value) = list.get(i) {
                            result.push(value.clone());
                        } else {
                            break;
                        }
                    }

                    Rc::new(Value::List(result))
                }
                Value::RangeList(range) => {
                    let mut range = range.borrow_mut();

                    let mut result = Vec::new();

                    for i in count..range.len() {
                        if let Some(value) = range.get(i) {
                            result.push(value.clone());
                        } else {
                            break;
                        }
                    }

                    Rc::new(Value::List(result))
                }
                _ => panic!("drop: invalid argument"),
            };

            self.stack.push(result);
            return;
        }

        panic!("drop: invalid argument");
    }

    fn std_rand(&mut self) {
        let max = self.stack.pop().unwrap();
        let min = self.stack.pop().unwrap();

        if let Value::Number(min) = min.as_ref() {
            if let Value::Number(max) = max.as_ref() {
                let min = *min as i32;
                let max = *max as i32;

                let value = self.rand_thread.gen_range(min..max);
                self.stack.push(Rc::new(Value::Number(value as f64)));
                return;
            }
        }

        panic!("rand: invalid argument");
    }

    pub fn interpret(&mut self, chunk: Block) -> VMResult {
        let frame = Frame {
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
        let mut last_line = 0;

        loop {
            if let Some(op) = self.read_byte() {
                let op = Opcode::from(op);

                if DEBUG {
                    let frame = self.frames.last().unwrap();
                    let closure = frame.get_closure().borrow();
                    let chunk = &closure.chunk;
                    let pc = frame.pc;

                    let mut data = EcoString::from(format!(".. .. .. {:02x}", op as u8));

                    let instruction = match op {
                        Opcode::LoadConstant => {
                            if let Some(idx) = chunk.read(pc) {
                                if let Some(value) = self.constants.get(*idx as usize) {
                                    data = format!(".. .. {:02x} {:02x}", op as u8, idx).into();

                                    format!("LD_CONST {:06x} = {:?}", idx, value)
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

                            if let Some(value) = self.constants.get(idx) {
                                format!("LD_CONST {:06x} = {:?}", idx, value)
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

                                let upvalue = closure.upvalues.get(*idx as usize).unwrap();
                                let value = upvalue.value.as_ref().unwrap();

                                format!("LD_UPVAL {:02x} (CLOSED) = {:?}", idx, value)
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
                        Opcode::LoadClosure => {
                            let idx = if let Some(idx) = chunk.read(pc) {
                                *idx as usize
                            } else {
                                panic!("missing function id");
                            };

                            let value = if let Some(value) = self.constants.get(idx) {
                                value.clone()
                            } else {
                                panic!("unknown function");
                            };

                            data = format!(".. .. {:02x} {:02x}", op as u8, idx).into();

                            format!("LD_CLOSURE {:02x} = {}", idx, value)
                        }
                        Opcode::LoadList => {
                            let size_lo = chunk.read(pc);
                            let size_hi = chunk.read(pc + 1);

                            let size = if let (Some(size_lo), Some(size_hi)) = (size_lo, size_hi) {
                                data =
                                    format!(".. {:02x} {:02x} {:02x}", op as u8, size_lo, size_hi)
                                        .into();

                                (*size_lo as usize) | ((*size_hi as usize) << 8)
                            } else {
                                panic!("missing list size");
                            };

                            format!("LD_LIST {:04x}", size)
                        }
                        Opcode::LoadRecord => {
                            let size_lo = chunk.read(pc);
                            let size_hi = chunk.read(pc + 1);

                            let size = if let (Some(size_lo), Some(size_hi)) = (size_lo, size_hi) {
                                data =
                                    format!(".. {:02x} {:02x} {:02x}", op as u8, size_lo, size_hi)
                                        .into();

                                (*size_lo as usize) | ((*size_hi as usize) << 8)
                            } else {
                                panic!("missing record size");
                            };

                            format!("LD_RECORD {:04x}", size)
                        }
                        Opcode::LoadRangeInclusive => "LD_RANGE_INC".to_string(),
                        Opcode::LoadRangeExclusive => "LD_RANGE_EXC".to_string(),
                        Opcode::Indexing => "INDEX".to_string(),
                        Opcode::LoadIterator => "LD_ITER".to_string(),
                        Opcode::LoopIterator => {
                            let end_lo = chunk.read(pc);
                            let end_md = chunk.read(pc + 1);
                            let end_hi = chunk.read(pc + 2);

                            let end = if let (Some(end_lo), Some(end_md), Some(end_hi)) =
                                (end_lo, end_md, end_hi)
                            {
                                data = format!(
                                    "{:02x} {:02x} {:02x} {:02x}",
                                    op as u8, end_lo, end_md, end_hi
                                )
                                .into();

                                (*end_lo as usize)
                                    | ((*end_md as usize) << 8)
                                    | ((*end_hi as usize) << 16)
                            } else {
                                panic!("missing jump offset");
                            };

                            format!("LOOP_ITER {:06x} ({:04x})", end, pc + end)
                        }
                        Opcode::IteratorAppend => "ITER_APPEND".to_string(),
                        Opcode::BuildString => {
                            let size = chunk.read(pc);

                            if let Some(size) = size {
                                data = format!(".. .. {:02x} {:02x}", op as u8, size).into();

                                format!("BUILD_STR {:02x}", size)
                            } else {
                                panic!("missing string size");
                            }
                        }
                        Opcode::Pop => "POP".to_string(),
                        Opcode::PopTop => "POP_TOP".to_string(),
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

                            format!("JMP {:06x} ({:04x})", offset, pc + offset)
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

                            format!("JF {:06x} ({:04x})", offset, pc + offset)
                        }
                        Opcode::JumpBack => {
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

                            format!("JB {:06x} ({:04x})", offset, pc - offset)
                        }
                        Opcode::Equals => "EQ".to_string(),
                        Opcode::NotEquals => "NEQ".to_string(),
                        Opcode::LessThan => "LT".to_string(),
                        Opcode::GreaterThan => "GT".to_string(),
                        Opcode::LessThanOrEqual => "LTE".to_string(),
                        Opcode::GreaterThanOrEqual => "GTE".to_string(),
                        Opcode::Negate => "NEG".to_string(),
                        Opcode::Add => "ADD".to_string(),
                        Opcode::Multiply => "MUL".to_string(),
                        Opcode::Divide => "DIV".to_string(),
                        Opcode::Subtract => "SUB".to_string(),
                        Opcode::Concatenate => "CONCAT".to_string(),
                        Opcode::Return => "RET".to_string(),
                        _ => todo!("unknown opcode: {:?}", op),
                    };

                    let line = chunk.get_line(pc - 1).unwrap();
                    let line = if line == last_line {
                        "    ".to_string()
                    } else {
                        last_line = line;
                        format!("{:>4}", line)
                    };

                    println!("{:04x}  {} | {}  {}", pc - 1, line, data, instruction);
                }

                match op {
                    Opcode::LoadConstant => self.op_ld_const(false)?,
                    Opcode::LoadConstantLong => self.op_ld_const(true)?,
                    Opcode::StoreGlobal => self.op_st_global()?,
                    Opcode::LoadLocal => self.op_ld_local()?,
                    Opcode::LoadGlobal => self.op_ld_global()?,
                    Opcode::LoadUpvalue => self.op_ld_upval()?,
                    Opcode::LoadTrue => self.stack.push(self.true_value.clone()),
                    Opcode::LoadFalse => self.stack.push(self.false_value.clone()),
                    Opcode::LoadZero => self.stack.push(self.zero_value.clone()),
                    Opcode::LoadOne => self.stack.push(self.one_value.clone()),
                    Opcode::LoadTwo => self.stack.push(self.two_value.clone()),
                    Opcode::LoadMinusOne => self.stack.push(self.minus_one_value.clone()),
                    Opcode::Call => self.op_call()?,
                    Opcode::LoadClosure => self.op_gen_closure()?,
                    Opcode::LoadList => self.op_ld_list()?,
                    Opcode::LoadRecord => self.op_ld_record()?,
                    Opcode::LoadRangeInclusive => self.op_ld_range(false)?,
                    Opcode::LoadRangeExclusive => self.op_ld_range(true)?,
                    Opcode::Indexing => self.op_index()?,
                    Opcode::LoadIterator => self.op_ld_iter()?,
                    Opcode::LoopIterator => self.op_loop_iter()?,
                    Opcode::IteratorAppend => self.op_iter_append()?,
                    Opcode::BuildString => self.op_build_str()?,
                    Opcode::Pop => self.op_pop()?,
                    Opcode::PopTop => self.op_pop_top()?,
                    Opcode::Jump => self.op_jmp()?,
                    Opcode::JumpIfFalse => self.op_jf()?,
                    Opcode::JumpBack => self.op_jb()?,
                    Opcode::Equals => self.op_eq()?,
                    Opcode::NotEquals => self.op_neq()?,
                    Opcode::LessThan => self.op_lt()?,
                    Opcode::GreaterThan => self.op_gt()?,
                    Opcode::LessThanOrEqual => self.op_lte()?,
                    Opcode::GreaterThanOrEqual => self.op_gte()?,
                    Opcode::Negate => self.op_neg()?,
                    Opcode::Add => self.op_add()?,
                    Opcode::Subtract => self.op_sub()?,
                    Opcode::Multiply => self.op_mul()?,
                    Opcode::Divide => self.op_div()?,
                    Opcode::Concatenate => self.op_concat()?,
                    Opcode::Return => {
                        let frame = self.frames.pop().unwrap();

                        if self.frames.is_empty() {
                            let result = self.stack.pop().unwrap_or(self.nil_value.clone());

                            if DEBUG {
                                println!("\n>>> RESULT = {:?}", result);

                                println!(">>> STACK");
                                for value in &self.stack {
                                    println!("{:?}", value);
                                }
                            }

                            return Ok(());
                        }

                        // The stack is in the following state:
                        // [...]
                        // foo   <- offset
                        // [...]
                        // bar   <- return value
                        //
                        // Everything between the offset and the return value is garbage, since
                        // any upvalues have been closed already. We can safely clean it up.
                        self.stack.drain(frame.offset..self.stack.len() - 1);
                    }
                    Opcode::Nop => {}
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
                    println!("callee: {:?}", callee);
                    panic!(
                        "call_value: expected {} arguments, got {}",
                        closure.arity, argc
                    );
                }

                // println!("{}", closure.chunk);

                let frame = Frame {
                    closure: value.clone(),
                    pc: 0,
                    offset: (self.stack.len() as isize - argc as isize) as usize,
                };

                self.frames.push(frame);
            }
            Value::NativeFunction { function, arity } => {
                if argc != *arity {
                    panic!("call_value: expected {} arguments, got {}", arity, argc);
                }

                // Safety: `self` is guaranteed to be valid for the lifetime of the program.
                // Also, I have no idea what I'm doing, but it works. I think. ¯\_(ツ)_/¯
                unsafe {
                    let self_ref = self as *mut VM;
                    function(&mut *self_ref);
                }
            }
            _ => {
                if argc > 0 {
                    panic!("call_value: value is not callable ({})", callee);
                }

                self.stack.push(callee.clone());
            }
        }
    }

    fn index_list(&mut self, list: Rc<Value>, index: Rc<Value>) -> VMResult {
        let index = if let Value::Number(index) = index.as_ref() {
            *index as usize
        } else {
            println!("index: invalid index, number expected");
            return Err(VMError::RuntimeError);
        };

        let value = match list.as_ref() {
            Value::List(items) => {
                if let Some(value) = items.get(index) {
                    value.clone()
                } else {
                    println!("index: out of bounds");
                    return Err(VMError::RuntimeError);
                }
            }
            Value::RangeList(range) => {
                if let Some(value) = range.borrow_mut().get(index) {
                    value.clone()
                } else {
                    println!("index: out of bounds");
                    return Err(VMError::RuntimeError);
                }
            }
            _ => unreachable!(),
        };

        self.stack.push(value);

        Ok(())
    }

    fn index_record(&mut self, record: Rc<Value>, key: Rc<Value>) -> VMResult {
        let key = if let Value::String(value) = key.as_ref() {
            value.clone()
        } else {
            println!("index: invalid key, string expected");
            return Err(VMError::RuntimeError);
        };

        let value = match record.as_ref() {
            Value::Record(entries) => {
                if let Some(value) = entries.get(&key) {
                    value.clone()
                } else {
                    println!("index: key \"{}\" not found in record", key);
                    return Err(VMError::RuntimeError);
                }
            }
            _ => unreachable!(),
        };

        self.stack.push(value);

        Ok(())
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

        if let Some(constant) = self.constants.get(id as usize) {
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

            // println!("idx = {} (offset: {})", idx, frame.offset);
            // println!("upvalues: {:#?}", closure.upvalues);

            if let Some(upvalue) = closure.upvalues.get(idx as usize) {
                let upvalue = upvalue.as_ref();
                let value = upvalue.value.as_ref().unwrap().clone();

                self.stack.push(value);
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
            println!("call: callee expected");
            return Err(VMError::RuntimeError);
        };

        // println!(">>> CALL {}:{}", callee, argc);

        self.call_value(&callee, argc);

        return Ok(());
    }

    // LD_CLOSURE <id: u8>
    fn op_gen_closure(&mut self) -> VMResult {
        let closure_id = if let Some(closure_id) = self.read_byte() {
            closure_id as usize
        } else {
            println!("gen_closure: missing constant id");
            return Err(VMError::RuntimeError);
        };

        let value = {
            if let Some(value) = self.constants.get(closure_id) {
                value.as_ref()
            } else {
                println!("gen_closure: unknown constant");
                return Err(VMError::RuntimeError);
            }
        };

        if let Value::Closure(closure) = value {
            let closure = &mut closure.borrow_mut();

            if !closure.upvalues.is_empty() {
                closure.upvalues.clear();
            }

            let frame = self.frames.last().unwrap();
            let parent_closure = if let Ok(parent_closure) = frame.get_closure().try_borrow_mut() {
                parent_closure.clone()
            } else {
                // If the parent closure is already borrowed, it means we're inside a recursive
                // function call.
                closure.clone()
            };

            for i in 0..closure.upvalue_refs.len() {
                let upvalue_ref = &closure.upvalue_refs[i];

                if upvalue_ref.is_local {
                    let index = frame.offset + upvalue_ref.index as usize;

                    let value = self.stack[index].clone();

                    if DEBUG {
                        println!(
                            "*                         CLOSE {:03} (offset: {}) = {:?}",
                            upvalue_ref.index as u8, frame.offset, value
                        );
                    }

                    let upvalue = Rc::new(Upvalue {
                        slot: index,
                        value: Some(value),
                    });
                    closure.upvalues.push(upvalue);
                } else {
                    if DEBUG {
                        println!(
                            "*                         REF {:03} = {}",
                            upvalue_ref.index as u8,
                            parent_closure
                                .upvalues
                                .get(upvalue_ref.index as usize)
                                .unwrap()
                                .value
                                .clone()
                                .unwrap(),
                        );
                    }

                    let u = parent_closure
                        .upvalues
                        .get(upvalue_ref.index as usize)
                        .unwrap();
                    closure.upvalues.push(u.clone());
                }
            }

            self.stack.pop();

            let closure = self.constants.get(closure_id).unwrap().clone();
            self.stack.push(closure);

            return Ok(());
        } else {
            println!("gen_closure: invalid closure");
            Err(VMError::RuntimeError)
        }
    }

    // LD_LIST <size: u16>
    fn op_ld_list(&mut self) -> VMResult {
        let size_lo = self.read_byte();
        let size_hi = self.read_byte();

        let size = if let (Some(size_lo), Some(size_hi)) = (size_lo, size_hi) {
            (size_lo as usize) | ((size_hi as usize) << 8)
        } else {
            println!("ld_list: missing size");
            return Err(VMError::RuntimeError);
        };

        let mut list = Vec::with_capacity(size);
        for _ in 0..size {
            if let Some(value) = self.stack.pop() {
                list.push(value);
            } else {
                println!("ld_list: missing value");
                return Err(VMError::RuntimeError);
            }
        }

        self.stack.push(Value::List(list.into()).into());

        Ok(())
    }

    // LD_RECORD <size: u16>
    fn op_ld_record(&mut self) -> VMResult {
        let size_lo = self.read_byte();
        let size_hi = self.read_byte();

        let size = if let (Some(size_lo), Some(size_hi)) = (size_lo, size_hi) {
            (size_lo as usize) | ((size_hi as usize) << 8)
        } else {
            println!("ld_record: missing size");
            return Err(VMError::RuntimeError);
        };

        let mut record = FxHashMap::<EcoString, Rc<Value>>::default();
        record.reserve(size);

        for _ in 0..size {
            let key = if let Some(key) = self.stack.pop() {
                if let Value::String(key) = key.as_ref() {
                    key.clone()
                } else {
                    println!("ld_record: key should be a string (this is probably a bug)");
                    return Err(VMError::RuntimeError);
                }
            } else {
                println!("ld_record: missing key");
                return Err(VMError::RuntimeError);
            };

            let value = if let Some(value) = self.stack.pop() {
                value
            } else {
                println!("ld_record: missing value");
                return Err(VMError::RuntimeError);
            };

            record.insert(key, value);
        }

        self.stack.push(Rc::new(Value::Record(record)));

        Ok(())
    }

    // LD_RANGE
    fn op_ld_range(&mut self, is_exclusive: bool) -> VMResult {
        let step = self.stack.pop();
        let end = self.stack.pop();
        let start = self.stack.pop();

        if let (Some(start), Some(end), Some(step)) = (start, end, step) {
            if let (Value::Number(start), Value::Number(end), Value::Number(step)) =
                (start.as_ref(), end.as_ref(), step.as_ref())
            {
                let end = if is_exclusive { end - 1.0 } else { *end };

                let range = Rc::new(RefCell::new(RangeListInner::new(
                    *start as usize,
                    end as usize,
                    *step as usize,
                )));
                self.stack.push(Value::RangeList(range).into());

                return Ok(());
            }
        }

        println!("ld_range: invalid range");
        Err(VMError::RuntimeError)
    }

    // INDEX
    fn op_index(&mut self) -> VMResult {
        let index = if let Some(index) = self.stack.pop() {
            index
        } else {
            println!("index: missing index");
            return Err(VMError::RuntimeError);
        };

        let value = if let Some(value) = self.stack.pop() {
            value
        } else {
            println!("index: missing value");
            return Err(VMError::RuntimeError);
        };

        match value.as_ref() {
            Value::List(_) | Value::RangeList(_) => self.index_list(value, index)?,
            Value::Record(_) => self.index_record(value, index)?,
            _ => {
                println!("index: invalid value, expected list or record");
                return Err(VMError::RuntimeError);
            }
        }

        Ok(())
    }

    // LD_ITER
    fn op_ld_iter(&mut self) -> VMResult {
        let collection = self.stack.pop();

        if let Some(collection) = collection {
            let iterator = Rc::new(RefCell::new(IteratorInner::new(collection.clone())));
            self.stack.push(Value::Iterator(iterator).into());

            return Ok(());
        }

        println!("load_iterator: missing collection");
        Err(VMError::RuntimeError)
    }

    // LOOP_ITER <offset: u24>
    fn op_loop_iter(&mut self) -> VMResult {
        let end_lo = self.read_byte();
        let end_md = self.read_byte();
        let end_hi = self.read_byte();

        let end = if let (Some(end_lo), Some(end_md), Some(end_hi)) = (end_lo, end_md, end_hi) {
            (end_lo as usize) | ((end_md as usize) << 8) | ((end_hi as usize) << 16)
        } else {
            println!("loop_iterator: missing end offset");
            return Err(VMError::RuntimeError);
        };

        let iterator = self.stack.pop();
        if let Some(iterator_obj) = iterator {
            if let Value::Iterator(iterator) = iterator_obj.as_ref() {
                let mut iterator = iterator.borrow_mut();

                if let Some(value) = iterator.next() {
                    self.stack.push(iterator_obj.clone());
                    self.stack.push(value);

                    return Ok(());
                }

                let frame = self.frames.last_mut().unwrap();
                frame.pc += end;

                let result = Rc::new(Value::List(iterator.result.clone()));
                self.stack.push(result);

                return Ok(());
            }

            println!("loop_iterator: invalid iterator");
            return Err(VMError::RuntimeError);
        }

        println!("loop_iterator: missing iterator");
        Err(VMError::RuntimeError)
    }

    // ITER_APPEND <offset: u24>
    fn op_iter_append(&mut self) -> VMResult {
        let offset_lo = self.read_byte();
        let offset_md = self.read_byte();
        let offset_hi = self.read_byte();

        let offset = if let (Some(offset_lo), Some(offset_md), Some(offset_hi)) =
            (offset_lo, offset_md, offset_hi)
        {
            (offset_lo as usize) | ((offset_md as usize) << 8) | ((offset_hi as usize) << 16)
        } else {
            println!("iter_append: missing offset");
            return Err(VMError::RuntimeError);
        };

        let value = self.stack.pop();
        let _id = self.stack.pop();
        let iterator = self.stack.get(0);

        if let (Some(iterator), Some(value)) = (iterator, value) {
            if let Value::Iterator(iterator) = iterator.as_ref() {
                let mut iterator = iterator.borrow_mut();
                iterator.push_result(value);

                let frame = self.frames.last_mut().unwrap();
                // println!("iter_append: jump to {}", frame.pc - offset);
                frame.pc -= offset;

                return Ok(());
            }

            println!("iter_st: invalid iterator");
            return Err(VMError::RuntimeError);
        }

        println!("iter_st: missing iterator or value");
        Err(VMError::RuntimeError)
    }

    // BUILD_STR <size: u8>
    fn op_build_str(&mut self) -> VMResult {
        let size = if let Some(size) = self.read_byte() {
            size as usize
        } else {
            println!("build_str: missing size");
            return Err(VMError::RuntimeError);
        };

        let mut result = EcoString::new();

        for _ in 0..size {
            let value = self.stack.pop().unwrap();
            let value = match value.as_ref() {
                Value::String(value) => value.to_string(),
                _ => value.to_string().to_string(),
            };

            result.push_str(&value);
        }

        self.stack.push(Value::String(result).into());

        Ok(())
    }

    // POP
    fn op_pop(&mut self) -> VMResult {
        if self.stack.pop().is_some() {
            return Ok(());
        }

        println!("pop: stack underflow");
        Err(VMError::RuntimeError)
    }

    // POP_TOP <idx: u8>
    fn op_pop_top(&mut self) -> VMResult {
        let frame = self.frames.last_mut().unwrap();

        let value = self.stack.remove(frame.offset);

        self.stack.push(value);

        return Ok(());
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

        if let Some(condition) = self.stack.pop() {
            if let Value::Bool(condition) = condition.as_ref() {
                if !condition {
                    let frame = self.frames.last_mut().unwrap();
                    frame.pc += offset;
                }

                return Ok(());
            }
        }

        println!("jf: invalid condition");
        Err(VMError::RuntimeError)
    }

    // JB <offset: u24>
    fn op_jb(&mut self) -> VMResult {
        let offset_lo = self.read_byte();
        let offset_md = self.read_byte();
        let offset_hi = self.read_byte();

        let offset = if let (Some(offset_lo), Some(offset_md), Some(offset_hi)) =
            (offset_lo, offset_md, offset_hi)
        {
            (offset_lo as usize) | ((offset_md as usize) << 8) | ((offset_hi as usize) << 16)
        } else {
            println!("jb: missing offset");
            return Err(VMError::RuntimeError);
        };

        let frame = self.frames.last_mut().unwrap();
        frame.pc -= offset;

        Ok(())
    }

    // EQ
    fn op_eq(&mut self) -> VMResult {
        let b = self.stack.pop();
        let a = self.stack.pop();

        if let (Some(a), Some(b)) = (a, b) {
            let result = match a == b {
                true => self.true_value.clone(),
                false => self.false_value.clone(),
            };

            self.stack.push(result);
            return Ok(());
        }

        println!("eq: invalid operands");
        Err(VMError::RuntimeError)
    }

    // NEQ
    fn op_neq(&mut self) -> VMResult {
        let b = self.stack.pop();
        let a = self.stack.pop();

        if let (Some(a), Some(b)) = (a, b) {
            let result = match a != b {
                true => self.true_value.clone(),
                false => self.false_value.clone(),
            };

            self.stack.push(result);
            return Ok(());
        }

        println!("neq: invalid operands");
        Err(VMError::RuntimeError)
    }

    // LT
    fn op_lt(&mut self) -> VMResult {
        let b = self.stack.pop();
        let a = self.stack.pop();

        if let (Some(a), Some(b)) = (a, b) {
            if let (Value::Number(a), Value::Number(b)) = (a.as_ref(), b.as_ref()) {
                let result = match a < b {
                    true => self.true_value.clone(),
                    false => self.false_value.clone(),
                };

                self.stack.push(result);
                return Ok(());
            }
        }

        println!("lt: invalid operands");
        Err(VMError::RuntimeError)
    }

    // GT
    fn op_gt(&mut self) -> VMResult {
        let b = self.stack.pop();
        let a = self.stack.pop();

        if let (Some(a), Some(b)) = (a, b) {
            if let (Value::Number(a), Value::Number(b)) = (a.as_ref(), b.as_ref()) {
                let result = match a > b {
                    true => self.true_value.clone(),
                    false => self.false_value.clone(),
                };

                self.stack.push(result);
                return Ok(());
            }
        }

        println!("gt: invalid operands");
        Err(VMError::RuntimeError)
    }

    // LTE
    fn op_lte(&mut self) -> VMResult {
        let b = self.stack.pop();
        let a = self.stack.pop();

        if let (Some(a), Some(b)) = (a, b) {
            if let (Value::Number(a), Value::Number(b)) = (a.as_ref(), b.as_ref()) {
                let result = match a <= b {
                    true => self.true_value.clone(),
                    false => self.false_value.clone(),
                };

                self.stack.push(result);
                return Ok(());
            }
        }

        println!("lte: invalid operands");
        Err(VMError::RuntimeError)
    }

    // GTE
    fn op_gte(&mut self) -> VMResult {
        let b = self.stack.pop();
        let a = self.stack.pop();

        if let (Some(a), Some(b)) = (a, b) {
            if let (Value::Number(a), Value::Number(b)) = (a.as_ref(), b.as_ref()) {
                let result = match a >= b {
                    true => self.true_value.clone(),
                    false => self.false_value.clone(),
                };

                self.stack.push(result);
                return Ok(());
            }
        }

        println!("gte: invalid operands");
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
                let result = (a + b) as usize;
                self.stack.push(Value::Number(result as f64).into());
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
            match a.as_ref() {
                Value::String(a) => {
                    let mut result = a.clone();
                    result.push_str(&b.to_string());

                    self.stack.push(Value::String(result.into()).into());

                    return Ok(());
                }
                Value::List(a) => {
                    let mut result = (*a).clone();

                    match b.as_ref() {
                        Value::List(b) => {
                            for value in b.iter() {
                                result.push(value.clone());
                            }
                        }
                        Value::RangeList(b) => {
                            b.borrow_mut().resolve();

                            for value in b.borrow().values.iter() {
                                result.push(value.clone());
                            }
                        }
                        _ => unreachable!(),
                    }

                    self.stack.push(Value::List(result.into()).into());

                    return Ok(());
                }
                Value::RangeList(a) => {
                    let result = match b.as_ref() {
                        Value::List(b) => {
                            let mut result = Vec::new();
                            result.extend(a.borrow_mut().clone());

                            for value in b.iter() {
                                result.push(value.clone());
                            }

                            result
                        }
                        Value::RangeList(b) => {
                            let mut result = Vec::new();

                            a.borrow_mut().resolve();
                            b.borrow_mut().resolve();

                            for value in a.borrow().values.iter() {
                                result.push(value.clone());
                            }

                            for value in b.borrow().values.iter() {
                                result.push(value.clone());
                            }

                            result
                        }
                        _ => unreachable!(),
                    };

                    self.stack.push(Rc::new(Value::List(result)));

                    return Ok(());
                }
                _ => {}
            }

            return Ok(());
        }

        println!("concat: invalid operands");
        Err(VMError::RuntimeError)
    }
}
