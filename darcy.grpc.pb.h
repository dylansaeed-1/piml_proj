// Generated by the gRPC C++ plugin.
// If you make any local change, they will be lost.
// source: darcy.proto
#ifndef GRPC_darcy_2eproto__INCLUDED
#define GRPC_darcy_2eproto__INCLUDED

#include "darcy.pb.h"

#include <functional>
#include <grpcpp/generic/async_generic_service.h>
#include <grpcpp/support/async_stream.h>
#include <grpcpp/support/async_unary_call.h>
#include <grpcpp/support/client_callback.h>
#include <grpcpp/client_context.h>
#include <grpcpp/completion_queue.h>
#include <grpcpp/support/message_allocator.h>
#include <grpcpp/support/method_handler.h>
#include <grpcpp/impl/proto_utils.h>
#include <grpcpp/impl/rpc_method.h>
#include <grpcpp/support/server_callback.h>
#include <grpcpp/impl/server_callback_handlers.h>
#include <grpcpp/server_context.h>
#include <grpcpp/impl/service_type.h>
#include <grpcpp/support/status.h>
#include <grpcpp/support/stub_options.h>
#include <grpcpp/support/sync_stream.h>

// The request message containing the user's name.
class piml final {
 public:
  static constexpr char const* service_full_name() {
    return "piml";
  }
  class StubInterface {
   public:
    virtual ~StubInterface() {}
    // Sends a greeting
    virtual ::grpc::Status get_residual(::grpc::ClientContext* context, const ::Array& request, ::Array* response) = 0;
    std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::Array>> Asyncget_residual(::grpc::ClientContext* context, const ::Array& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::Array>>(Asyncget_residualRaw(context, request, cq));
    }
    std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::Array>> PrepareAsyncget_residual(::grpc::ClientContext* context, const ::Array& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::Array>>(PrepareAsyncget_residualRaw(context, request, cq));
    }
    class async_interface {
     public:
      virtual ~async_interface() {}
      // Sends a greeting
      virtual void get_residual(::grpc::ClientContext* context, const ::Array* request, ::Array* response, std::function<void(::grpc::Status)>) = 0;
      virtual void get_residual(::grpc::ClientContext* context, const ::Array* request, ::Array* response, ::grpc::ClientUnaryReactor* reactor) = 0;
    };
    typedef class async_interface experimental_async_interface;
    virtual class async_interface* async() { return nullptr; }
    class async_interface* experimental_async() { return async(); }
   private:
    virtual ::grpc::ClientAsyncResponseReaderInterface< ::Array>* Asyncget_residualRaw(::grpc::ClientContext* context, const ::Array& request, ::grpc::CompletionQueue* cq) = 0;
    virtual ::grpc::ClientAsyncResponseReaderInterface< ::Array>* PrepareAsyncget_residualRaw(::grpc::ClientContext* context, const ::Array& request, ::grpc::CompletionQueue* cq) = 0;
  };
  class Stub final : public StubInterface {
   public:
    Stub(const std::shared_ptr< ::grpc::ChannelInterface>& channel, const ::grpc::StubOptions& options = ::grpc::StubOptions());
    ::grpc::Status get_residual(::grpc::ClientContext* context, const ::Array& request, ::Array* response) override;
    std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::Array>> Asyncget_residual(::grpc::ClientContext* context, const ::Array& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::Array>>(Asyncget_residualRaw(context, request, cq));
    }
    std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::Array>> PrepareAsyncget_residual(::grpc::ClientContext* context, const ::Array& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::Array>>(PrepareAsyncget_residualRaw(context, request, cq));
    }
    class async final :
      public StubInterface::async_interface {
     public:
      void get_residual(::grpc::ClientContext* context, const ::Array* request, ::Array* response, std::function<void(::grpc::Status)>) override;
      void get_residual(::grpc::ClientContext* context, const ::Array* request, ::Array* response, ::grpc::ClientUnaryReactor* reactor) override;
     private:
      friend class Stub;
      explicit async(Stub* stub): stub_(stub) { }
      Stub* stub() { return stub_; }
      Stub* stub_;
    };
    class async* async() override { return &async_stub_; }

   private:
    std::shared_ptr< ::grpc::ChannelInterface> channel_;
    class async async_stub_{this};
    ::grpc::ClientAsyncResponseReader< ::Array>* Asyncget_residualRaw(::grpc::ClientContext* context, const ::Array& request, ::grpc::CompletionQueue* cq) override;
    ::grpc::ClientAsyncResponseReader< ::Array>* PrepareAsyncget_residualRaw(::grpc::ClientContext* context, const ::Array& request, ::grpc::CompletionQueue* cq) override;
    const ::grpc::internal::RpcMethod rpcmethod_get_residual_;
  };
  static std::unique_ptr<Stub> NewStub(const std::shared_ptr< ::grpc::ChannelInterface>& channel, const ::grpc::StubOptions& options = ::grpc::StubOptions());

  class Service : public ::grpc::Service {
   public:
    Service();
    virtual ~Service();
    // Sends a greeting
    virtual ::grpc::Status get_residual(::grpc::ServerContext* context, const ::Array* request, ::Array* response);
  };
  template <class BaseClass>
  class WithAsyncMethod_get_residual : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service* /*service*/) {}
   public:
    WithAsyncMethod_get_residual() {
      ::grpc::Service::MarkMethodAsync(0);
    }
    ~WithAsyncMethod_get_residual() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status get_residual(::grpc::ServerContext* /*context*/, const ::Array* /*request*/, ::Array* /*response*/) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    void Requestget_residual(::grpc::ServerContext* context, ::Array* request, ::grpc::ServerAsyncResponseWriter< ::Array>* response, ::grpc::CompletionQueue* new_call_cq, ::grpc::ServerCompletionQueue* notification_cq, void *tag) {
      ::grpc::Service::RequestAsyncUnary(0, context, request, response, new_call_cq, notification_cq, tag);
    }
  };
  typedef WithAsyncMethod_get_residual<Service > AsyncService;
  template <class BaseClass>
  class WithCallbackMethod_get_residual : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service* /*service*/) {}
   public:
    WithCallbackMethod_get_residual() {
      ::grpc::Service::MarkMethodCallback(0,
          new ::grpc::internal::CallbackUnaryHandler< ::Array, ::Array>(
            [this](
                   ::grpc::CallbackServerContext* context, const ::Array* request, ::Array* response) { return this->get_residual(context, request, response); }));}
    void SetMessageAllocatorFor_get_residual(
        ::grpc::MessageAllocator< ::Array, ::Array>* allocator) {
      ::grpc::internal::MethodHandler* const handler = ::grpc::Service::GetHandler(0);
      static_cast<::grpc::internal::CallbackUnaryHandler< ::Array, ::Array>*>(handler)
              ->SetMessageAllocator(allocator);
    }
    ~WithCallbackMethod_get_residual() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status get_residual(::grpc::ServerContext* /*context*/, const ::Array* /*request*/, ::Array* /*response*/) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    virtual ::grpc::ServerUnaryReactor* get_residual(
      ::grpc::CallbackServerContext* /*context*/, const ::Array* /*request*/, ::Array* /*response*/)  { return nullptr; }
  };
  typedef WithCallbackMethod_get_residual<Service > CallbackService;
  typedef CallbackService ExperimentalCallbackService;
  template <class BaseClass>
  class WithGenericMethod_get_residual : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service* /*service*/) {}
   public:
    WithGenericMethod_get_residual() {
      ::grpc::Service::MarkMethodGeneric(0);
    }
    ~WithGenericMethod_get_residual() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status get_residual(::grpc::ServerContext* /*context*/, const ::Array* /*request*/, ::Array* /*response*/) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
  };
  template <class BaseClass>
  class WithRawMethod_get_residual : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service* /*service*/) {}
   public:
    WithRawMethod_get_residual() {
      ::grpc::Service::MarkMethodRaw(0);
    }
    ~WithRawMethod_get_residual() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status get_residual(::grpc::ServerContext* /*context*/, const ::Array* /*request*/, ::Array* /*response*/) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    void Requestget_residual(::grpc::ServerContext* context, ::grpc::ByteBuffer* request, ::grpc::ServerAsyncResponseWriter< ::grpc::ByteBuffer>* response, ::grpc::CompletionQueue* new_call_cq, ::grpc::ServerCompletionQueue* notification_cq, void *tag) {
      ::grpc::Service::RequestAsyncUnary(0, context, request, response, new_call_cq, notification_cq, tag);
    }
  };
  template <class BaseClass>
  class WithRawCallbackMethod_get_residual : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service* /*service*/) {}
   public:
    WithRawCallbackMethod_get_residual() {
      ::grpc::Service::MarkMethodRawCallback(0,
          new ::grpc::internal::CallbackUnaryHandler< ::grpc::ByteBuffer, ::grpc::ByteBuffer>(
            [this](
                   ::grpc::CallbackServerContext* context, const ::grpc::ByteBuffer* request, ::grpc::ByteBuffer* response) { return this->get_residual(context, request, response); }));
    }
    ~WithRawCallbackMethod_get_residual() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status get_residual(::grpc::ServerContext* /*context*/, const ::Array* /*request*/, ::Array* /*response*/) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    virtual ::grpc::ServerUnaryReactor* get_residual(
      ::grpc::CallbackServerContext* /*context*/, const ::grpc::ByteBuffer* /*request*/, ::grpc::ByteBuffer* /*response*/)  { return nullptr; }
  };
  template <class BaseClass>
  class WithStreamedUnaryMethod_get_residual : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service* /*service*/) {}
   public:
    WithStreamedUnaryMethod_get_residual() {
      ::grpc::Service::MarkMethodStreamed(0,
        new ::grpc::internal::StreamedUnaryHandler<
          ::Array, ::Array>(
            [this](::grpc::ServerContext* context,
                   ::grpc::ServerUnaryStreamer<
                     ::Array, ::Array>* streamer) {
                       return this->Streamedget_residual(context,
                         streamer);
                  }));
    }
    ~WithStreamedUnaryMethod_get_residual() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable regular version of this method
    ::grpc::Status get_residual(::grpc::ServerContext* /*context*/, const ::Array* /*request*/, ::Array* /*response*/) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    // replace default version of method with streamed unary
    virtual ::grpc::Status Streamedget_residual(::grpc::ServerContext* context, ::grpc::ServerUnaryStreamer< ::Array,::Array>* server_unary_streamer) = 0;
  };
  typedef WithStreamedUnaryMethod_get_residual<Service > StreamedUnaryService;
  typedef Service SplitStreamedService;
  typedef WithStreamedUnaryMethod_get_residual<Service > StreamedService;
};


#endif  // GRPC_darcy_2eproto__INCLUDED
