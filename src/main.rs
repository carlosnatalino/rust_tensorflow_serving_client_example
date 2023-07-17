use std::sync::Arc;
use std::collections::HashMap;

use grpcio::{Environment, ChannelBuilder};
use protobuf::RepeatedField;

use tensorflow_serving_client::tensor::TensorProto;
use tensorflow_serving_client::tensor_shape::TensorShapeProto_Dim;

use tensorflow_serving_client::model::ModelSpec;
use tensorflow_serving_client::prediction_service_grpc::PredictionServiceClient;
use tensorflow_serving_client::predict::PredictRequest;
use tensorflow_serving_client::types::DataType;

use linfa_datasets;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {

    // let input_vec = vec![2.3, 5.1, 2.3, 5.4];
    let dataset = linfa_datasets::iris();
    // println!("Samples: {}", dataset.ntargets());

    // let s = dataset.one_vs_all()?;
    // let records = s[0].1.records();

    // for i in 0..records.len() {
    //     println!("{} -> {:?}", i, records.row(i));
    //     let input: f32 = records.row(i).into_iter().map(|f| f.to_owned()).collect::<f32>();
    // }
    
    // for (i, record) in dataset.one_vs_all()?[0].1.records().enumerate() {
    //     println!("{}: {:?}", i, record.);
    // }

    let mut request = PredictRequest::new();
    
    let mut model_spec = ModelSpec::new();
    model_spec.set_name("iris".to_string());
    request.set_model_spec(model_spec);
    
    let env = Environment::new(5);
    let channel = ChannelBuilder::new(Arc::new(env)).connect("localhost:8500");
    let client = PredictionServiceClient::new(channel);
    
    for record in dataset.one_vs_all().unwrap() {
        let arr = record.1.records();
        for i in 0..(arr.len() / 4) {

            let x: Vec<f32> = arr.row(i).into_iter().map(|f| f.to_owned() as f32).collect();

            let mut dim_1 = TensorShapeProto_Dim::new();
            dim_1.set_size(1);
            
            let mut dim_2 = TensorShapeProto_Dim::new();
            dim_2.set_size(x.len() as i64);

            let mut rf: RepeatedField<TensorShapeProto_Dim> = RepeatedField::new();
            rf.push(dim_1);
            rf.push(dim_2);

            let mut tp = TensorProto::new();
            tp.set_dtype(DataType::DT_FLOAT);

            tp.mut_tensor_shape().set_dim(rf);

            println!("\n{}", i);

            println!("\t{:?}", x);
            // println!("\t{:?}", &input);
            tp.set_float_val(x);
    
            let mut rm: HashMap<String, TensorProto> = HashMap::new();
            rm.insert("dense_4_input".to_string(), tp);

            request.set_inputs(rm);

            let result = client.predict(&request);

            if let Some(class) = result?.outputs.get("dense_7") {
                println!("\t{:?}", class.float_val);
            }
        }
    }
    
    Ok(())
}
