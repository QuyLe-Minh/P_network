import torch
import tensorrt as trt

from unetr_pp.rt_inference.inferencer import SegmentationNetwork


def convert_to_onnx(model, input_tensors, onnx_path='model.onnx'):
    model.eval()  # Important: Set the model to evaluation mode
    torch.onnx.export(
        model,
        input_tensors,
        onnx_path,
        input_names=["image", "pos"],
        output_names=["output"], 
        export_params=True,
        opset_version=13,  # TensorRT 8.5 supports up to opset 13
        do_constant_folding=True,  # Optimizes constant operations
        dynamic_axes=None  # Static shapes (better for TensorRT unless you specifically want dynamic)
    )
    print(f"ONNX export completed: {onnx_path}")

TRT_LOGGER = trt.Logger()

def build_engine(onnx_file_path, trt_file_path='model.trt', use_fp16=False):
    # initialize TensorRT engine and parse ONNX model
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
     
    # parse ONNX
    with open(onnx_file_path, 'rb') as model:
        print('Beginning ONNX file parsing')
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise RuntimeError("Failed to parse ONNX file")
    print('Completed parsing of ONNX file')
    
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
    if use_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        
    # generate TensorRT engine optimized for the target platform
    print('Building an engine...')
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Failed to build the engine")
    print("Completed creating Engine")
    
    # Save the engine to a file
    with open(trt_file_path, "wb") as f:
        f.write(serialized_engine)
    print(f"Engine saved to {trt_file_path}")
    return serialized_engine


class TensorRTModel(SegmentationNetwork):
    def __init__(self, trt_path):
        """
        Initialize the TensorRT model
        
        Args:
            trt_path: Path to the serialized TensorRT engine file
        """
        super().__init__()
        # Initialize TensorRT logger
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        # Load engine from file
        with open(trt_path, "rb") as f:
            engine_bytes = f.read()
        
        # Create runtime and deserialize engine
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_bytes)
        self.context = self.engine.create_execution_context()
        
        # Create CUDA stream
        self.stream = torch.cuda.Stream()
        
        # Pre-allocate structures for inputs and outputs
        self.input_names = []
        self.input_buffers = []
        self.output_names = []
        self.output_buffers = []
        self.buffer_dict = {}
        
        # Setup tensor information
        self._setup_io_tensors()
        
    def _setup_io_tensors(self):
        """Initialize input and output tensor information"""
        # Process all tensors
        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            is_input = self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT
            
            if is_input:
                self.input_names.append(tensor_name)
            else:
                self.output_names.append(tensor_name)
                
        # Print some debug info
        print(f"Engine has {len(self.input_names)} inputs and {len(self.output_names)} outputs")
    
    def _prepare_buffers(self, input_tensors):
        """
        Prepare input and output buffers for inference
        
        Args:
            input_tensors: List of PyTorch tensors for input (must be on CUDA)
        """
        # Clear previous buffers
        self.input_buffers = []
        self.output_buffers = []
        self.buffer_dict = {}
        
        # Process input tensors
        for i, tensor_name in enumerate(self.input_names):
            if i >= len(input_tensors):
                raise ValueError(f"More input tensors expected ({len(self.input_names)}) than provided ({len(input_tensors)})")
            
            tensor = input_tensors[i]
            # Set dynamic shape if needed
            shape = tuple(self.context.get_tensor_shape(tensor_name))
            if -1 in shape:
                self.context.set_tensor_shape(tensor_name, tensor.shape)
                shape = tuple(self.context.get_tensor_shape(tensor_name))
                
            # Determine data type
            trt_dtype = self.engine.get_tensor_dtype(tensor_name)
            if trt_dtype == trt.DataType.FLOAT:
                dtype = torch.float32
            elif trt_dtype == trt.DataType.HALF:
                dtype = torch.float16
            elif trt_dtype == trt.DataType.INT32:
                dtype = torch.int32
            else:
                dtype = torch.float32
                
            # Create buffer and convert input tensor if needed
            buffer = torch.empty(shape, dtype=dtype, device="cuda")
            if tensor.dtype != buffer.dtype:
                tensor = tensor.to(buffer.dtype)
            
            # Copy data to buffer
            buffer.copy_(tensor)
            
            # Store buffer
            self.input_buffers.append(buffer)
            self.buffer_dict[tensor_name] = buffer
            
        # Process output tensors
        for tensor_name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(tensor_name))
            trt_dtype = self.engine.get_tensor_dtype(tensor_name)
            
            if trt_dtype == trt.DataType.FLOAT:
                dtype = torch.float32
            elif trt_dtype == trt.DataType.HALF:
                dtype = torch.float16
            elif trt_dtype == trt.DataType.INT32:
                dtype = torch.int32
            else:
                dtype = torch.float32
                
            buffer = torch.empty(shape, dtype=dtype, device="cuda")
            self.output_buffers.append(buffer)
            self.buffer_dict[tensor_name] = buffer
            
    def forward(self, *input_tensors):
        """
        Run inference with the TensorRT engine
        
        Args:
            *input_tensors: PyTorch tensors for input (must be on CUDA)
            
        Returns:
            List of numpy arrays containing the inference results
        """
        # Check inputs for NaN values
        for i, tensor in enumerate(input_tensors):
            if torch.isnan(tensor).any():
                print(f"Warning: Input tensor {i} contains NaN values!")
            if torch.isinf(tensor).any():
                print(f"Warning: Input tensor {i} contains infinity values!")
        
        # Prepare buffers for this inference
        self._prepare_buffers(input_tensors)
        
        # Set input tensors
        for tensor_name in self.input_names:
            self.context.set_tensor_address(tensor_name, self.buffer_dict[tensor_name].data_ptr())
        
        # Set output tensors
        for tensor_name in self.output_names:
            self.context.set_tensor_address(tensor_name, self.buffer_dict[tensor_name].data_ptr())
        
        # Execute inference
        with torch.cuda.stream(self.stream):
            success = self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
            if not success:
                print("Warning: TensorRT execution reported failure")
        
        # Synchronize CUDA stream
        self.stream.synchronize()
        
        # Check output for NaN values
        results = []
        for i, buffer in enumerate(self.output_buffers):
            if torch.isnan(buffer).any():
                print(f"Warning: Output buffer {i} contains NaN values!")
                print(f"Shape: {buffer.shape}, dtype: {buffer.dtype}")
            
            # Convert to numpy
            result = buffer.cpu().numpy()
            results.append(result)
            
        return torch.from_numpy(results[0])
        
    def __del__(self):
        """Clean up resources"""
        if hasattr(self, 'context'):
            del self.context
        if hasattr(self, 'engine'):
            del self.engine
        if hasattr(self, 'runtime'):
            del self.runtime


def run_inference_with_tensorrt(trt_path, input_tensors):
    """
    Run inference with a TensorRT engine
    
    Args:
        trt_path: Path to the serialized TensorRT engine file
        input_tensors: List of PyTorch tensors for input (must be on CUDA)
        
    Returns:
        List of numpy arrays containing the inference results
    """
    # Check inputs for NaN values
    for i, tensor in enumerate(input_tensors):
        if torch.isnan(tensor).any():
            print(f"Warning: Input tensor {i} contains NaN values!")
        if torch.isinf(tensor).any():
            print(f"Warning: Input tensor {i} contains infinity values!")
    
    # Initialize TensorRT logger
    logger = trt.Logger(trt.Logger.WARNING)
    
    # Load engine from file
    with open(trt_path, "rb") as f:
        engine_bytes = f.read()
    
    # Create runtime and deserialize engine
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    context = engine.create_execution_context()
    
    try:
        # Create CUDA stream
        stream = torch.cuda.Stream()
        
        # Prepare inputs and outputs using the newer TensorRT API
        inputs = []
        outputs = []
        output_buffers = []
        
        # Debug info about engine
        # print(f"Engine has {engine.num_io_tensors} IO tensors")
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            mode = "INPUT" if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT else "OUTPUT"
            dtype = engine.get_tensor_dtype(name)
            # print(f"  Tensor {i}: {name} ({mode}) - Type: {dtype}")
        
        # Process all tensors
        for i in range(engine.num_io_tensors):
            tensor_name = engine.get_tensor_name(i)
            is_input = engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT
            
            # Get shape and handle dynamic dimensions
            shape = tuple(context.get_tensor_shape(tensor_name))
            if is_input and -1 in shape:
                input_idx = len(inputs)
                if input_idx < len(input_tensors):
                    context.set_tensor_shape(tensor_name, input_tensors[input_idx].shape)
                    shape = tuple(context.get_tensor_shape(tensor_name))
                    print(f"Set dynamic shape for {tensor_name}: {shape}")
            
            # Determine data type
            trt_dtype = engine.get_tensor_dtype(tensor_name)
            if trt_dtype == trt.DataType.FLOAT:
                dtype = torch.float32
            elif trt_dtype == trt.DataType.HALF:
                dtype = torch.float16
                print(f"Warning: Using FP16 for tensor {tensor_name}")
            elif trt_dtype == trt.DataType.INT32:
                dtype = torch.int32
            else:
                dtype = torch.float32
                print(f"Using default float32 for unknown type: {trt_dtype}")
            
            # Create buffer
            buffer = torch.empty(shape, dtype=dtype, device="cuda")
            
            # Set input or collect output
            if is_input:
                inputs.append((tensor_name, buffer))
            else:
                outputs.append(tensor_name)
                output_buffers.append(buffer)
        
        # Copy input data to device and set input tensors
        for i, tensor in enumerate(input_tensors):
            if i >= len(inputs):
                raise ValueError(f"More input tensors provided ({len(input_tensors)}) than engine expects ({len(inputs)})")
            
            tensor_name, buffer = inputs[i]
            # Check shapes
            if tensor.shape != buffer.shape:
                raise ValueError(f"Input tensor shape {tensor.shape} doesn't match expected shape {buffer.shape}")
            
            # Convert dtype if necessary
            if tensor.dtype != buffer.dtype:
                print(f"Converting input tensor {i} from {tensor.dtype} to {buffer.dtype}")
                tensor = tensor.to(buffer.dtype)
            
            # Copy data to input buffer
            buffer.copy_(tensor)
            
            # Set input tensor for execution context
            context.set_tensor_address(tensor_name, buffer.data_ptr())
        
        # Set output tensors
        for i, tensor_name in enumerate(outputs):
            context.set_tensor_address(tensor_name, output_buffers[i].data_ptr())
        
        # Execute inference
        with torch.cuda.stream(stream):
            success = context.execute_async_v3(stream_handle=stream.cuda_stream)
            if not success:
                print("Warning: TensorRT execution reported failure")
        
        # Synchronize CUDA stream
        stream.synchronize()
        
        # Check output for NaN values
        results = []
        for i, buffer in enumerate(output_buffers):
            if torch.isnan(buffer).any():
                print(f"Warning: Output buffer {i} contains NaN values!")
                # Try fallback to FP32 calculation
                print(f"Shape: {buffer.shape}, dtype: {buffer.dtype}")
            
            # Convert to numpy
            result = buffer.cpu().numpy()
            results.append(result)
        return results
    
    finally:
        # Clean up resources
        del context
        del engine
        del runtime
