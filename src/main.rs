mod lexer;
// mod llm;
mod parser;
mod scanner;
mod vm;

#[allow(unused_imports)]
use std::time::Instant;

use codespan_reporting::{
    files::SimpleFiles,
    term::{
        self,
        termcolor::{ColorChoice, StandardStream},
    },
};
use line_numbers::LinePositions;

use crate::{
    lexer::{Lexer, ToDiagnostic},
    parser::Parser,
    vm::{block::Block, compiler::Compiler, VM},
};

const DEBUG: bool = false;
const TIME: bool = false;

fn main() {
    let mut start = Instant::now();
    let mut mem_before = 0;

    let source = std::fs::read_to_string("./examples/test.tail").unwrap();

    let mut files = SimpleFiles::new();
    files.add("src", &source);

    let line_positions = LinePositions::from(source.as_str());

    let writer = StandardStream::stderr(ColorChoice::Auto);
    let mut config = term::Config::default();
    config.chars = term::Chars::ascii();

    if TIME {
        start = Instant::now();
    }

    let mut lexer = Lexer::new(&source);
    if let Err(err) = lexer.lex() {
        let diagnostic = err.to_diagnostic();
        term::emit(&mut writer.lock(), &config, &files, &diagnostic).unwrap();
        return;
    }

    if TIME {
        println!("Lexer took:\t\t{:?}", Instant::now() - start);

        mem_before = if let Some(stats) = memory_stats::memory_stats() {
            stats.physical_mem + stats.virtual_mem
        } else {
            0
        };
        start = Instant::now();
    }

    // let mut lexer_copy = lexer.clone();

    let mut parser = Parser::new(lexer);
    let result = parser.parse();
    if let Err(err) = result {
        // println!("--- Tokens:");
        // loop {
        //     let token = lexer_copy.next();
        //     print!("{:#?}\n", token);

        //     if let lexer::Token::EOF { .. } = token {
        //         break;
        //     }
        // }

        let diagnostic = parser.error_to_diagnostic(err);
        term::emit(&mut writer.lock(), &config, &files, &diagnostic).unwrap();
        return;
    }

    let ast = result.unwrap();

    if TIME {
        let end = Instant::now();
        let mem_after = if let Some(stats) = memory_stats::memory_stats() {
            stats.physical_mem + stats.virtual_mem
        } else {
            0
        };

        println!(
            "Parser took:\t\t{:?} ({} bytes)",
            end - start,
            mem_after - mem_before
        );

        // println!("\n--- AST:");
        // println!("{:#?}", ast);

        mem_before = if let Some(stats) = memory_stats::memory_stats() {
            stats.physical_mem + stats.virtual_mem
        } else {
            0
        };
        start = Instant::now();
    }

    let mut block = Block::new();

    let mut compiler = Compiler::new(line_positions);
    if let Err(err) = compiler.compile(&ast, &mut block) {
        panic!("{:?}", err);
    }

    if TIME {
        let end = Instant::now();
        let mem_after = if let Some(stats) = memory_stats::memory_stats() {
            stats.physical_mem + stats.virtual_mem
        } else {
            0
        };

        println!(
            "Compiler took:\t\t{:?} ({} mb)",
            end - start,
            (mem_after - mem_before) / 1_000_000
        );
    }

    if DEBUG {
        println!("\n[PROGRAM]\n{}", block);

        println!("[CONSTANTS]");
        for (i, constant) in compiler.constants().iter().enumerate() {
            println!("{:06x}  {:?}", i, constant);
        }
        println!();
    }

    if TIME {
        mem_before = if let Some(stats) = memory_stats::memory_stats() {
            stats.physical_mem + stats.virtual_mem
        } else {
            0
        };
        start = Instant::now();
    }

    let mut vm = VM::new(compiler.constants().clone());
    vm.load_stdlib();

    if let Err(err) = vm.interpret(block) {
        panic!("{:?}", err);
    }

    if TIME {
        let end = Instant::now();
        let mem_after = if let Some(stats) = memory_stats::memory_stats() {
            stats.physical_mem + stats.virtual_mem
        } else {
            0
        };

        println!(
            "VM took:\t\t{:?} ({} mb)",
            end - start,
            (mem_after - mem_before) / 1_000_000
        );

        println!("Total memory used:\t{} mb", mem_after / 1_000_000);
    }
}
