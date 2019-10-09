import abc
import base64

import grpc
import requests
from tensorflow import make_tensor_proto, string
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc


class ModelConfig:
    def __init__(self, model_name, signature_name="serving_default"):
        self.model_name = model_name
        self.signature_name = signature_name


class _BaseClient(abc.ABC):
    def __init__(self, proto_coder, model_config: ModelConfig, host="localhost", port=8500):
        self.proto_coder = proto_coder
        self.host = host
        self.port = port
        self.model_config = model_config

    @abc.abstractmethod
    def predict(self, inputs: list, input_shape):
        pass


class BaseRESTClient(_BaseClient):

    def __init__(self, proto_coder, model_config, host="localhost", port=8501):
        super().__init__(proto_coder, model_config, host, port)
        self.server_url = "http://{host}:{port}/v1/models/{model_name}:predict". \
            format(host=host, port=port,
                   model_name=model_config.model_name)

    def predict(self, inputs, input_shape=None):
        prepared_inputs = self.prepare_inputs(inputs)
        return requests.post(
            self.server_url, data=prepared_inputs)

    def prepare_inputs(self, inputs):
        proto_coded_inputs = []
        for data in inputs:
            proto_coded_inputs.append(self.proto_coder.encode(data))
        prepared_inputs = []
        for proto_coded_input in proto_coded_inputs:
            encoded_input = base64.b64encode(proto_coded_input).decode('utf-8')
            predict_request = '{ "b64": "%s" }' % encoded_input
            prepared_inputs.append(predict_request)
        prepared_inputs = '{ "instances": [' + ','.join(map(str, prepared_inputs)) + ']}'
        return prepared_inputs


class BaseGRPCClient(_BaseClient):

    def __init__(self, proto_coder, model_config, host="localhost", port=8500):
        super().__init__(proto_coder, model_config, host, port)
        self.stub = None
        self.request = None
        self.initialize()

    def initialize(self):
        channel = grpc.insecure_channel('{host}:{port}'.format(host=self.host, port=self.port))
        self.stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
        self.request = self.GRPCRequest(signature_name=self.model_config.signature_name,
                                        model_name=self.model_config.model_name, proto_coder=self.proto_coder)

    def predict(self, inputs, input_shape):
        prepared_request = self.request.prepare_request_for(inputs, input_shape)
        return self.stub.Predict(prepared_request, 10.0)

    class GRPCRequest:
        def __init__(self, signature_name, model_name, proto_coder):
            self.signature_name = signature_name
            self.model_name = model_name
            self.proto_coder = proto_coder

        def prepare_request_for(self, inputs, input_shape) -> predict_pb2.PredictRequest:
            request = predict_pb2.PredictRequest()
            request.model_spec.signature_name = self.signature_name
            request.model_spec.name = self.model_name
            proto_coded_inputs = []
            for data in inputs:
                proto_coded_inputs.append(self.proto_coder.encode(data))
            request.inputs["examples"].CopyFrom(make_tensor_proto(proto_coded_inputs, dtype=string, shape=[1, ]))

            return request
