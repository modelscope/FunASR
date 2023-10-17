// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <fst/extensions/far/far-class.h>

#include <fst/script/script-impl.h>
#include <fst/extensions/far/script-impl.h>

namespace fst {
namespace script {


// FarReaderClass.

FarReaderClass *FarReaderClass::Open(const string &filename) {
  const std::vector<string> filenames{filename};
  return FarReaderClass::Open(filenames);
}

FarReaderClass *FarReaderClass::Open(const std::vector<string> &filenames) {
  if (filenames.empty()) {
    LOG(ERROR) << "FarReaderClass::Open: No files specified";
    return nullptr;
  }
  const auto arc_type = LoadArcTypeFromFar(filenames.front());
  if (arc_type.empty()) return nullptr;
  OpenFarReaderClassArgs args(filenames);
  args.retval = nullptr;
  Apply<Operation<OpenFarReaderClassArgs>>("OpenFarReaderClass", arc_type,
                                            &args);
  return args.retval;
}

REGISTER_FST_OPERATION(OpenFarReaderClass, StdArc, OpenFarReaderClassArgs);
REGISTER_FST_OPERATION(OpenFarReaderClass, LogArc, OpenFarReaderClassArgs);
REGISTER_FST_OPERATION(OpenFarReaderClass, Log64Arc, OpenFarReaderClassArgs);

// FarWriterClass.

FarWriterClass *FarWriterClass::Create(const string &filename,
                                       const string &arc_type, FarType type) {
  CreateFarWriterClassInnerArgs iargs(filename, type);
  CreateFarWriterClassArgs args(iargs);
  args.retval = nullptr;
  Apply<Operation<CreateFarWriterClassArgs>>("CreateFarWriterClass", arc_type,
                                             &args);
  return args.retval;
}

REGISTER_FST_OPERATION(CreateFarWriterClass, StdArc, CreateFarWriterClassArgs);
REGISTER_FST_OPERATION(CreateFarWriterClass, LogArc, CreateFarWriterClassArgs);
REGISTER_FST_OPERATION(CreateFarWriterClass, Log64Arc,
                       CreateFarWriterClassArgs);

}  // namespace script
}  // namespace fst
