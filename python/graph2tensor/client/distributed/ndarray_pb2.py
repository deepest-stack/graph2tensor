# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: ndarray.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='ndarray.proto',
  package='graph2tensor.client',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\rndarray.proto\x12\x13graph2tensor.client\"F\n\x0cNdarrayProto\x12\r\n\x05\x64type\x18\x01 \x01(\t\x12\x10\n\x04\x64ims\x18\x02 \x03(\x05\x42\x02\x10\x01\x12\x15\n\rarray_content\x18\x03 \x01(\x0c\x62\x06proto3'
)




_NDARRAYPROTO = _descriptor.Descriptor(
  name='NdarrayProto',
  full_name='graph2tensor.client.NdarrayProto',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='dtype', full_name='graph2tensor.client.NdarrayProto.dtype', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='dims', full_name='graph2tensor.client.NdarrayProto.dims', index=1,
      number=2, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=b'\020\001', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='array_content', full_name='graph2tensor.client.NdarrayProto.array_content', index=2,
      number=3, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=38,
  serialized_end=108,
)

DESCRIPTOR.message_types_by_name['NdarrayProto'] = _NDARRAYPROTO
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

NdarrayProto = _reflection.GeneratedProtocolMessageType('NdarrayProto', (_message.Message,), {
  'DESCRIPTOR' : _NDARRAYPROTO,
  '__module__' : 'ndarray_pb2'
  # @@protoc_insertion_point(class_scope:graph2tensor.client.NdarrayProto)
  })
_sym_db.RegisterMessage(NdarrayProto)


_NDARRAYPROTO.fields_by_name['dims']._options = None
# @@protoc_insertion_point(module_scope)