#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>

#include <grpc/grpc.h>
#include <grpcpp/security/server_credentials.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/server_context.h>
#ifdef BAZEL_BUILD
#include "examples/protos/route_guide.grpc.pb.h"
#else
#include "darcy.grpc.pb.h"
#endif

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerReader;
using grpc::ServerReaderWriter;
using grpc::ServerWriter;
using grpc::Status;
using std::chrono::system_clock;


class RouteGuideImpl final : public piml::Service {
 public:
  explicit RouteGuideImpl() {}

  Status get_residual(ServerContext* context, const Array* in, Array* out){
    std::cout << "called: " << in->n_cols() << std::endl;
    out->set_n_cols(10);
    out->set_n_rows(12);
    return Status::OK;
  }

};

void RunServer() {
  std::string server_address("0.0.0.0:50051");
  RouteGuideImpl service{};

  ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  std::unique_ptr<Server> server(builder.BuildAndStart());
  std::cout << "Server listening on " << server_address << std::endl;
  server->Wait();
}

int main(int argc, char** argv) {
  // Expect only arg: --db_path=path/to/route_guide_db.json.
  RunServer();

  return 0;
}
