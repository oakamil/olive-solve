// Required Notice: Copyright (c) 2026 Omair Kamil
// See LICENSE file in root directory for license terms.

use clap::Parser;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;
use tonic::transport::Server;

use tetra3::{extractor::CentroidConfig, extractor::Extractor, solver::Solver};
use tetra3_server::{Tetra3ServerImpl, proto::tetra3_service_server::Tetra3ServiceServer};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the Tetra3 database zip file
    #[arg(short, long)]
    database_path: PathBuf,

    /// gRPC port to listen on
    #[arg(short, long, default_value_t = 50051)]
    port: u16,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    println!("Loading database from: {:?}", args.database_path);
    let solver = Solver::load_database(&args.database_path)?;
    println!("Database loaded successfully.");

    let extractor = Extractor::new(CentroidConfig::default());

    let service = Tetra3ServerImpl {
        solver: Arc::new(Mutex::new(solver)),
        extractor: Arc::new(Mutex::new(extractor)),
    };

    let addr = format!("0.0.0.0:{}", args.port).parse()?;
    println!("Tetra3 gRPC server listening on {}", addr);

    Server::builder()
        .add_service(Tetra3ServiceServer::new(service))
        .serve(addr)
        .await?;

    Ok(())
}
