mod interpreter;
mod lexer;
mod llm;
mod parser;
mod scanner;

use std::{
    sync::{Arc, Mutex},
    time::Instant,
};

use codespan_reporting::{
    files::SimpleFiles,
    term::{
        self,
        termcolor::{ColorChoice, StandardStream},
    },
};

use crate::{
    interpreter::Environment,
    lexer::{Lexer, ToDiagnostic},
    parser::Parser,
};

fn main() {
    let source = std::fs::read_to_string("./examples/test.tail").unwrap();

    let mut files = SimpleFiles::new();
    files.add("src", &source);

    let writer = StandardStream::stderr(ColorChoice::Auto);
    let mut config = term::Config::default();
    config.chars = term::Chars::ascii();

    let start = Instant::now();

    let mut lexer = Lexer::new(&source);
    if let Err(err) = lexer.lex() {
        let diagnostic = err.to_diagnostic();
        term::emit(&mut writer.lock(), &config, &files, &diagnostic).unwrap();
        return;
    }

    println!("Lexer took:\t\t{:?}", Instant::now() - start);

    // let mut lexer_copy = lexer.clone();

    let mem_before = if let Some(stats) = memory_stats::memory_stats() {
        stats.physical_mem + stats.virtual_mem
    } else {
        0
    };
    let start = Instant::now();

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

    let mem_before = if let Some(stats) = memory_stats::memory_stats() {
        stats.physical_mem + stats.virtual_mem
    } else {
        0
    };
    let start = Instant::now();

    let mut environment = Arc::new(Mutex::new(Environment::new()));
    // environment.load_stdlib();

    let result = ast.eval(&mut environment);
    if let Err(err) = result {
        let diagnostic = err.to_diagnostic();
        term::emit(&mut writer.lock(), &config, &files, &diagnostic).unwrap();
        return;
    }

    let end = Instant::now();
    let mem_after = if let Some(stats) = memory_stats::memory_stats() {
        stats.physical_mem + stats.virtual_mem
    } else {
        0
    };

    println!(
        "Interpreter took:\t{:?} ({} mb)",
        end - start,
        (mem_after - mem_before) / 1_000_000
    );

    println!("Total memory used:\t{} mb", mem_after / 1_000_000);

    println!("\n{}", result.unwrap());
}
