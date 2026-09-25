//! Inter-device transfers. In-process channels stand in for cluster links;
//! an optional TCP fabric talks over localhost ports (one per node).

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use catnn_core::tensor::Tensor;
use catnn_core::{CatError, CatResult};

/// A named buffer moved across a link.
#[derive(Debug, Clone)]
pub struct TransferMsg {
    pub src_device: String,
    pub dst_device: String,
    pub tensor: Tensor,
    pub tag: String,
}

/// In-process fabric: one inbox per device.
#[derive(Clone)]
pub struct ChannelFabric {
    txs: Arc<HashMap<String, Sender<TransferMsg>>>,
    rxs: Arc<Mutex<HashMap<String, Receiver<TransferMsg>>>>,
}

impl ChannelFabric {
    pub fn new(devices: &[String]) -> Self {
        let mut txs = HashMap::new();
        let mut rxs = HashMap::new();
        for d in devices {
            let (tx, rx) = mpsc::channel();
            txs.insert(d.clone(), tx);
            rxs.insert(d.clone(), rx);
        }
        Self { txs: Arc::new(txs), rxs: Arc::new(Mutex::new(rxs)) }
    }

    pub fn send(&self, msg: TransferMsg) -> CatResult<()> {
        let tx = self.txs.get(&msg.dst_device).ok_or_else(|| {
            CatError::Io(format!("no inbox for {}", msg.dst_device))
        })?;
        tx.send(msg).map_err(|e| CatError::Io(e.to_string()))
    }

    pub fn take_rx(&self, device: &str) -> Option<Receiver<TransferMsg>> {
        self.rxs.lock().unwrap().remove(device)
    }
}

/// Very small length-prefixed tensor protocol over TCP.
pub fn spawn_tcp_node(bind: &str) -> CatResult<(u16, thread::JoinHandle<()>, Sender<TransferMsg>)> {
    let listener = TcpListener::bind(bind).map_err(|e| CatError::Io(e.to_string()))?;
    listener.set_nonblocking(false).ok();
    let port = listener.local_addr().map_err(|e| CatError::Io(e.to_string()))?.port();
    let (tx, rx) = mpsc::channel::<TransferMsg>();
    let handle = thread::spawn(move || {
        if let Ok((mut stream, _)) = listener.accept() {
            let _ = write_msg(&mut stream, &rx.recv().unwrap());
        }
    });
    Ok((port, handle, tx))
}

pub fn send_tcp(addr: &str, msg: &TransferMsg) -> CatResult<()> {
    let mut stream = TcpStream::connect(addr).map_err(|e| CatError::Io(e.to_string()))?;
    stream
        .set_write_timeout(Some(Duration::from_secs(2)))
        .ok();
    write_msg(&mut stream, msg)
}

fn write_msg(stream: &mut TcpStream, msg: &TransferMsg) -> CatResult<()> {
    let payload = serde_json::to_vec(&(
        &msg.src_device,
        &msg.dst_device,
        &msg.tag,
        &msg.tensor.data,
        msg.tensor.ty.shape.dims.clone(),
    ))
    .map_err(|e| CatError::Io(e.to_string()))?;
    let len = (payload.len() as u32).to_le_bytes();
    stream.write_all(&len).map_err(|e| CatError::Io(e.to_string()))?;
    stream.write_all(&payload).map_err(|e| CatError::Io(e.to_string()))?;
    Ok(())
}

pub fn read_msg(stream: &mut TcpStream) -> CatResult<TransferMsg> {
    let mut lenb = [0u8; 4];
    stream.read_exact(&mut lenb).map_err(|e| CatError::Io(e.to_string()))?;
    let len = u32::from_le_bytes(lenb) as usize;
    let mut buf = vec![0u8; len];
    stream.read_exact(&mut buf).map_err(|e| CatError::Io(e.to_string()))?;
    let (src, dst, tag, data, dims): (String, String, String, Vec<f32>, Vec<usize>) =
        serde_json::from_slice(&buf).map_err(|e| CatError::Io(e.to_string()))?;
    let tensor = Tensor::from_vec(catnn_core::tensor::TensorType::f32(dims), data)?;
    Ok(TransferMsg { src_device: src, dst_device: dst, tensor, tag })
}

#[cfg(test)]
mod tests {
    use super::*;
    use catnn_core::tensor::TensorType;

    #[test]
    fn channel_roundtrip() {
        let f = ChannelFabric::new(&["a".into(), "b".into()]);
        let rx = f.take_rx("b").unwrap();
        f.send(TransferMsg {
            src_device: "a".into(),
            dst_device: "b".into(),
            tensor: Tensor::zeros(TensorType::f32([2])),
            tag: "t".into(),
        })
        .unwrap();
        let m = rx.recv().unwrap();
        assert_eq!(m.tag, "t");
    }
}
