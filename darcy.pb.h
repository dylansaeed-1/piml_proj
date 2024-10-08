// Generated by the protocol buffer compiler.  DO NOT EDIT!
// NO CHECKED-IN PROTOBUF GENCODE
// source: darcy.proto
// Protobuf C++ Version: 5.27.2

#ifndef GOOGLE_PROTOBUF_INCLUDED_darcy_2eproto_2epb_2eh
#define GOOGLE_PROTOBUF_INCLUDED_darcy_2eproto_2epb_2eh

#include <limits>
#include <string>
#include <type_traits>
#include <utility>

#include "google/protobuf/runtime_version.h"
#if PROTOBUF_VERSION != 5027002
#error "Protobuf C++ gencode is built with an incompatible version of"
#error "Protobuf C++ headers/runtime. See"
#error "https://protobuf.dev/support/cross-version-runtime-guarantee/#cpp"
#endif
#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/arena.h"
#include "google/protobuf/arenastring.h"
#include "google/protobuf/generated_message_tctable_decl.h"
#include "google/protobuf/generated_message_util.h"
#include "google/protobuf/metadata_lite.h"
#include "google/protobuf/generated_message_reflection.h"
#include "google/protobuf/message.h"
#include "google/protobuf/repeated_field.h"  // IWYU pragma: export
#include "google/protobuf/extension_set.h"  // IWYU pragma: export
#include "google/protobuf/unknown_field_set.h"
// @@protoc_insertion_point(includes)

// Must be included last.
#include "google/protobuf/port_def.inc"

#define PROTOBUF_INTERNAL_EXPORT_darcy_2eproto

namespace google {
namespace protobuf {
namespace internal {
class AnyMetadata;
}  // namespace internal
}  // namespace protobuf
}  // namespace google

// Internal implementation detail -- do not use these members.
struct TableStruct_darcy_2eproto {
  static const ::uint32_t offsets[];
};
extern const ::google::protobuf::internal::DescriptorTable
    descriptor_table_darcy_2eproto;
class Array;
struct ArrayDefaultTypeInternal;
extern ArrayDefaultTypeInternal _Array_default_instance_;
namespace google {
namespace protobuf {
}  // namespace protobuf
}  // namespace google


// ===================================================================


// -------------------------------------------------------------------

class Array final : public ::google::protobuf::Message
/* @@protoc_insertion_point(class_definition:Array) */ {
 public:
  inline Array() : Array(nullptr) {}
  ~Array() override;
  template <typename = void>
  explicit PROTOBUF_CONSTEXPR Array(
      ::google::protobuf::internal::ConstantInitialized);

  inline Array(const Array& from) : Array(nullptr, from) {}
  inline Array(Array&& from) noexcept
      : Array(nullptr, std::move(from)) {}
  inline Array& operator=(const Array& from) {
    CopyFrom(from);
    return *this;
  }
  inline Array& operator=(Array&& from) noexcept {
    if (this == &from) return *this;
    if (GetArena() == from.GetArena()
#ifdef PROTOBUF_FORCE_COPY_IN_MOVE
        && GetArena() != nullptr
#endif  // !PROTOBUF_FORCE_COPY_IN_MOVE
    ) {
      InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }

  inline const ::google::protobuf::UnknownFieldSet& unknown_fields() const
      ABSL_ATTRIBUTE_LIFETIME_BOUND {
    return _internal_metadata_.unknown_fields<::google::protobuf::UnknownFieldSet>(::google::protobuf::UnknownFieldSet::default_instance);
  }
  inline ::google::protobuf::UnknownFieldSet* mutable_unknown_fields()
      ABSL_ATTRIBUTE_LIFETIME_BOUND {
    return _internal_metadata_.mutable_unknown_fields<::google::protobuf::UnknownFieldSet>();
  }

  static const ::google::protobuf::Descriptor* descriptor() {
    return GetDescriptor();
  }
  static const ::google::protobuf::Descriptor* GetDescriptor() {
    return default_instance().GetMetadata().descriptor;
  }
  static const ::google::protobuf::Reflection* GetReflection() {
    return default_instance().GetMetadata().reflection;
  }
  static const Array& default_instance() {
    return *internal_default_instance();
  }
  static inline const Array* internal_default_instance() {
    return reinterpret_cast<const Array*>(
        &_Array_default_instance_);
  }
  static constexpr int kIndexInFileMessages = 0;
  friend void swap(Array& a, Array& b) { a.Swap(&b); }
  inline void Swap(Array* other) {
    if (other == this) return;
#ifdef PROTOBUF_FORCE_COPY_IN_SWAP
    if (GetArena() != nullptr && GetArena() == other->GetArena()) {
#else   // PROTOBUF_FORCE_COPY_IN_SWAP
    if (GetArena() == other->GetArena()) {
#endif  // !PROTOBUF_FORCE_COPY_IN_SWAP
      InternalSwap(other);
    } else {
      ::google::protobuf::internal::GenericSwap(this, other);
    }
  }
  void UnsafeArenaSwap(Array* other) {
    if (other == this) return;
    ABSL_DCHECK(GetArena() == other->GetArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  Array* New(::google::protobuf::Arena* arena = nullptr) const final {
    return ::google::protobuf::Message::DefaultConstruct<Array>(arena);
  }
  using ::google::protobuf::Message::CopyFrom;
  void CopyFrom(const Array& from);
  using ::google::protobuf::Message::MergeFrom;
  void MergeFrom(const Array& from) { Array::MergeImpl(*this, from); }

  private:
  static void MergeImpl(
      ::google::protobuf::MessageLite& to_msg,
      const ::google::protobuf::MessageLite& from_msg);

  public:
  bool IsInitialized() const {
    return true;
  }
  ABSL_ATTRIBUTE_REINITIALIZES void Clear() final;
  ::size_t ByteSizeLong() const final;
  ::uint8_t* _InternalSerialize(
      ::uint8_t* target,
      ::google::protobuf::io::EpsCopyOutputStream* stream) const final;
  int GetCachedSize() const { return _impl_._cached_size_.Get(); }

  private:
  void SharedCtor(::google::protobuf::Arena* arena);
  void SharedDtor();
  void InternalSwap(Array* other);
 private:
  friend class ::google::protobuf::internal::AnyMetadata;
  static ::absl::string_view FullMessageName() { return "Array"; }

 protected:
  explicit Array(::google::protobuf::Arena* arena);
  Array(::google::protobuf::Arena* arena, const Array& from);
  Array(::google::protobuf::Arena* arena, Array&& from) noexcept
      : Array(arena) {
    *this = ::std::move(from);
  }
  const ::google::protobuf::Message::ClassData* GetClassData() const final;

 public:
  ::google::protobuf::Metadata GetMetadata() const;
  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------
  enum : int {
    kValFieldNumber = 3,
    kNColsFieldNumber = 1,
    kNRowsFieldNumber = 2,
  };
  // repeated double val = 3;
  int val_size() const;
  private:
  int _internal_val_size() const;

  public:
  void clear_val() ;
  double val(int index) const;
  void set_val(int index, double value);
  void add_val(double value);
  const ::google::protobuf::RepeatedField<double>& val() const;
  ::google::protobuf::RepeatedField<double>* mutable_val();

  private:
  const ::google::protobuf::RepeatedField<double>& _internal_val() const;
  ::google::protobuf::RepeatedField<double>* _internal_mutable_val();

  public:
  // int32 n_cols = 1;
  void clear_n_cols() ;
  ::int32_t n_cols() const;
  void set_n_cols(::int32_t value);

  private:
  ::int32_t _internal_n_cols() const;
  void _internal_set_n_cols(::int32_t value);

  public:
  // int32 n_rows = 2;
  void clear_n_rows() ;
  ::int32_t n_rows() const;
  void set_n_rows(::int32_t value);

  private:
  ::int32_t _internal_n_rows() const;
  void _internal_set_n_rows(::int32_t value);

  public:
  // @@protoc_insertion_point(class_scope:Array)
 private:
  class _Internal;
  friend class ::google::protobuf::internal::TcParser;
  static const ::google::protobuf::internal::TcParseTable<
      2, 3, 0,
      0, 2>
      _table_;

  static constexpr const void* _raw_default_instance_ =
      &_Array_default_instance_;

  friend class ::google::protobuf::MessageLite;
  friend class ::google::protobuf::Arena;
  template <typename T>
  friend class ::google::protobuf::Arena::InternalHelper;
  using InternalArenaConstructable_ = void;
  using DestructorSkippable_ = void;
  struct Impl_ {
    inline explicit constexpr Impl_(
        ::google::protobuf::internal::ConstantInitialized) noexcept;
    inline explicit Impl_(::google::protobuf::internal::InternalVisibility visibility,
                          ::google::protobuf::Arena* arena);
    inline explicit Impl_(::google::protobuf::internal::InternalVisibility visibility,
                          ::google::protobuf::Arena* arena, const Impl_& from,
                          const Array& from_msg);
    ::google::protobuf::RepeatedField<double> val_;
    ::int32_t n_cols_;
    ::int32_t n_rows_;
    mutable ::google::protobuf::internal::CachedSize _cached_size_;
    PROTOBUF_TSAN_DECLARE_MEMBER
  };
  union { Impl_ _impl_; };
  friend struct ::TableStruct_darcy_2eproto;
};

// ===================================================================




// ===================================================================


#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// -------------------------------------------------------------------

// Array

// int32 n_cols = 1;
inline void Array::clear_n_cols() {
  ::google::protobuf::internal::TSanWrite(&_impl_);
  _impl_.n_cols_ = 0;
}
inline ::int32_t Array::n_cols() const {
  // @@protoc_insertion_point(field_get:Array.n_cols)
  return _internal_n_cols();
}
inline void Array::set_n_cols(::int32_t value) {
  _internal_set_n_cols(value);
  // @@protoc_insertion_point(field_set:Array.n_cols)
}
inline ::int32_t Array::_internal_n_cols() const {
  ::google::protobuf::internal::TSanRead(&_impl_);
  return _impl_.n_cols_;
}
inline void Array::_internal_set_n_cols(::int32_t value) {
  ::google::protobuf::internal::TSanWrite(&_impl_);
  _impl_.n_cols_ = value;
}

// int32 n_rows = 2;
inline void Array::clear_n_rows() {
  ::google::protobuf::internal::TSanWrite(&_impl_);
  _impl_.n_rows_ = 0;
}
inline ::int32_t Array::n_rows() const {
  // @@protoc_insertion_point(field_get:Array.n_rows)
  return _internal_n_rows();
}
inline void Array::set_n_rows(::int32_t value) {
  _internal_set_n_rows(value);
  // @@protoc_insertion_point(field_set:Array.n_rows)
}
inline ::int32_t Array::_internal_n_rows() const {
  ::google::protobuf::internal::TSanRead(&_impl_);
  return _impl_.n_rows_;
}
inline void Array::_internal_set_n_rows(::int32_t value) {
  ::google::protobuf::internal::TSanWrite(&_impl_);
  _impl_.n_rows_ = value;
}

// repeated double val = 3;
inline int Array::_internal_val_size() const {
  return _internal_val().size();
}
inline int Array::val_size() const {
  return _internal_val_size();
}
inline void Array::clear_val() {
  ::google::protobuf::internal::TSanWrite(&_impl_);
  _impl_.val_.Clear();
}
inline double Array::val(int index) const {
  // @@protoc_insertion_point(field_get:Array.val)
  return _internal_val().Get(index);
}
inline void Array::set_val(int index, double value) {
  _internal_mutable_val()->Set(index, value);
  // @@protoc_insertion_point(field_set:Array.val)
}
inline void Array::add_val(double value) {
  ::google::protobuf::internal::TSanWrite(&_impl_);
  _internal_mutable_val()->Add(value);
  // @@protoc_insertion_point(field_add:Array.val)
}
inline const ::google::protobuf::RepeatedField<double>& Array::val() const
    ABSL_ATTRIBUTE_LIFETIME_BOUND {
  // @@protoc_insertion_point(field_list:Array.val)
  return _internal_val();
}
inline ::google::protobuf::RepeatedField<double>* Array::mutable_val()
    ABSL_ATTRIBUTE_LIFETIME_BOUND {
  // @@protoc_insertion_point(field_mutable_list:Array.val)
  ::google::protobuf::internal::TSanWrite(&_impl_);
  return _internal_mutable_val();
}
inline const ::google::protobuf::RepeatedField<double>&
Array::_internal_val() const {
  ::google::protobuf::internal::TSanRead(&_impl_);
  return _impl_.val_;
}
inline ::google::protobuf::RepeatedField<double>* Array::_internal_mutable_val() {
  ::google::protobuf::internal::TSanRead(&_impl_);
  return &_impl_.val_;
}

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif  // __GNUC__

// @@protoc_insertion_point(namespace_scope)


// @@protoc_insertion_point(global_scope)

#include "google/protobuf/port_undef.inc"

#endif  // GOOGLE_PROTOBUF_INCLUDED_darcy_2eproto_2epb_2eh
