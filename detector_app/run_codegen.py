from grpc_tools import protoc

protoc.main((
    '',
    '-Iproto',
    '--python_out=./grpc_inference',
    '--grpc_python_out=./grpc_inference',
    '--pyi_out=./grpc_inference',
    'proto/inference.proto',
))
