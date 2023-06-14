// util/kaldi-io.h

// Copyright 2009-2011  Microsoft Corporation;  Jan Silovsky
//                2016  Xiaohui Zhang

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//  http://www.apache.org/licenses/LICENSE-2.0

// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.
#ifndef KALDI_UTIL_KALDI_IO_H_
#define KALDI_UTIL_KALDI_IO_H_

#ifdef _MSC_VER
# include <fcntl.h>
# include <io.h>
#endif
#include <cctype>  // For isspace.
#include <limits>
#include <string>
#include "base/kaldi-common.h"
#include "matrix/kaldi-matrix.h"


namespace kaldi {

class OutputImplBase;  // Forward decl; defined in a .cc file
class InputImplBase;  // Forward decl; defined in a .cc file

/// \addtogroup io_group
/// @{

// The Output and Input classes handle stream-opening for "extended" filenames
// that include actual files, standard-input/standard-output, pipes, and
// offsets into actual files.  They also handle reading and writing the
// binary-mode headers for Kaldi files, where applicable.  The classes have
// versions of the Open routines that throw and do not throw, depending whether
// the calling code wants to catch the errors or not; there are also versions
// that write (or do not write) the Kaldi binary-mode header that says if it's
// binary mode.  Generally files that contain Kaldi objects will have the header
// on, so we know upon reading them whether they have the header.  So you would
// use the OpenWithHeader routines for these (or the constructor); but other
// types of objects (e.g. FSTs) would have files without a header so you would
// use OpenNoHeader.

// We now document the types of extended filenames that we use.
//
// A "wxfilename"  is an extended filename for writing. It can take three forms:
// (1) Filename: e.g.    "/some/filename", "./a/b/c", "c:\Users\dpovey\My
//                        Documents\\boo"
//          (whatever the actual file-system interprets)
// (2) Standard output:  "" or "-"
// (3) A pipe: e.g.  "gunzip -c /tmp/abc.gz |"
//
//
// A "rxfilename" is an extended filename for reading.  It can take four forms:
// (1) An actual filename, whatever the file-system can read, e.g. "/my/file".
// (2) Standard input: "" or "-"
// (3) A pipe: e.g. "| gzip -c > /tmp/abc.gz"
// (4) An offset into a file, e.g.: "/mnt/blah/data/1.ark:24871"
//   [these are created by the Table and TableWriter classes; I may also write
//    a program that creates them for arbitrary files]
//


// Typical usage:
// ...
// bool binary;
// MyObject.Write(Output(some_filename, binary).Stream(), binary);
//
// ... more extensive example:
// {
//    Output ko(some_filename, binary);
//    MyObject1.Write(ko.Stream(), binary);
//    MyObject2.Write(ko.Stream(), binary);
// }



enum OutputType {
  kNoOutput,
  kFileOutput,
  kStandardOutput,
  kPipeOutput
};

/// ClassifyWxfilename interprets filenames as follows:
///  - kNoOutput: invalid filenames (leading or trailing space, things that look
///     like wspecifiers and rspecifiers or like pipes to read from with leading
///     |.
///  - kFileOutput: Normal filenames
///  - kStandardOutput: The empty string or "-", interpreted as standard output
///  - kPipeOutput: pipes, e.g. "gunzip -c some_file.gz |"
OutputType ClassifyWxfilename(const std::string &wxfilename);

enum InputType {
  kNoInput,
  kFileInput,
  kStandardInput,
  kOffsetFileInput,
  kPipeInput
};

/// ClassifyRxfilenames interprets filenames for reading as follows:
///  - kNoInput: invalid filenames (leading or trailing space, things that
///       look like wspecifiers and rspecifiers or pipes to write to
///       with trailing |.
///  - kFileInput: normal filenames
///  - kStandardInput: the empty string or "-"
///  - kPipeInput: e.g. "| gzip -c > blah.gz"
///  - kOffsetFileInput: offsets into files, e.g.  /some/filename:12970
InputType ClassifyRxfilename(const std::string &rxfilename);


class Output {
 public:
  // The normal constructor, provided for convenience.
  // Equivalent to calling with default constructor then Open()
  // with these arguments.
  Output(const std::string &filename, bool binary, bool write_header = true);

  Output(): impl_(NULL) {}

  /// This opens the stream, with the given mode (binary or text).  It returns
  /// true on success and false on failure.  However, it will throw if something
  /// was already open and could not be closed (to avoid this, call Close()
  /// first.  if write_header == true and binary == true, it writes the Kaldi
  /// binary-mode header ('\0' then 'B').  You may call Open even if it is
  /// already open; it will close the existing stream and reopen (however if
  /// closing the old stream failed it will throw).
  bool Open(const std::string &wxfilename, bool binary, bool write_header);

  inline bool IsOpen();  // return true if we have an open stream.  Does not
  // imply stream is good for writing.

  std::ostream &Stream();  // will throw if not open; else returns stream.

  // Close closes the stream. Calling Close is never necessary unless you
  // want to avoid exceptions being thrown.  There are times when calling
  // Close will hurt efficiency (basically, when using offsets into files,
  // and using the same Input object),
  // but most of the time the user won't be doing this directly, it will
  // be done in kaldi-table.{h, cc}, so you don't have to worry about it.
  bool Close();

  // This will throw if stream could not be closed (to check error status,
  // call Close()).
  ~Output();

 private:
  OutputImplBase *impl_;  // non-NULL if open.
  std::string filename_;
  KALDI_DISALLOW_COPY_AND_ASSIGN(Output);
};


// bool binary_in;
// Input ki(some_filename, &binary_in);
// MyObject.Read(ki, binary_in);
//
// ... more extensive example:
//
// {
//    bool binary_in;
//    Input ki(some_filename, &binary_in);
//    MyObject1.Read(ki.Stream(), &binary_in);
//    MyObject2.Write(ki.Stream(), &binary_in);
// }
// Note that to catch errors you need to use try.. catch.
// Input communicates errors by throwing exceptions.


// Input interprets four kinds of filenames:
//  (1) Normal filenames
//  (2) The empty string or "-", interpreted as standard output
//  (3) Pipes, e.g. "| gzip -c > some_file.gz"
//  (4) Offsets into [real] files, e.g. "/my/filename:12049"
// The last one has no correspondence in Output.


class Input {
 public:
  /// The normal constructor.  Opens the stream in binary mode.
  /// Equivalent to calling the default constructor followed by Open(); then, if
  /// binary != NULL, it calls ReadHeader(), putting the output in "binary"; it
  /// throws on error.
  Input(const std::string &rxfilename, bool *contents_binary = NULL);

  Input(): impl_(NULL) {}

  // Open opens the stream for reading (the mode, where relevant, is binary; use
  // OpenTextMode for text-mode, we made this a separate function rather than a
  // boolean argument, to avoid confusion with Kaldi's text/binary distinction,
  // since reading in the file system's text mode is unusual.)  If
  // contents_binary != NULL, it reads the binary-mode header and puts it in the
  // "binary" variable.  Returns true on success.  If it returns false it will
  // not be open.  You may call Open even if it is already open; it will close
  // the existing stream and reopen (however if closing the old stream failed it
  // will throw).
  inline bool Open(const std::string &rxfilename, bool *contents_binary = NULL);

  // As Open but (if the file system has text/binary modes) opens in text mode;
  // you shouldn't ever have to use this as in Kaldi we read even text files in
  // binary mode (and ignore the \r).
  inline bool OpenTextMode(const std::string &rxfilename);

  // Return true if currently open for reading and Stream() will
  // succeed.  Does not guarantee that the stream is good.
  inline bool IsOpen();

  // It is never necessary or helpful to call Close, except if
  // you are concerned about to many filehandles being open.
  // Close does not throw. It returns the exit code as int32
  // in the case of a pipe [kPipeInput], and always zero otherwise.
  int32 Close();

  // Returns the underlying stream. Throws if !IsOpen()
  std::istream &Stream();

  // Destructor does not throw: input streams may legitimately fail so we
  // don't worry about the status when we close them.
  ~Input();
 private:
  bool OpenInternal(const std::string &rxfilename, bool file_binary,
                    bool *contents_binary);
  InputImplBase *impl_;
  KALDI_DISALLOW_COPY_AND_ASSIGN(Input);
};

template <class C> void ReadKaldiObject(const std::string &filename,
                                        C *c) {
  bool binary_in;
  Input ki(filename, &binary_in);
  c->Read(ki.Stream(), binary_in);
}

// Specialize the template for reading matrices, because we want to be able to
// support reading 'ranges' (row and column ranges), like foo.mat[10:20].
template <> void ReadKaldiObject(const std::string &filename,
                                 Matrix<float> *m);


template <> void ReadKaldiObject(const std::string &filename,
                                 Matrix<double> *m);



template <class C> inline void WriteKaldiObject(const C &c,
                                                const std::string &filename,
                                                bool binary) {
  Output ko(filename, binary);
  c.Write(ko.Stream(), binary);
}

/// PrintableRxfilename turns the rxfilename into a more human-readable
/// form for error reporting, i.e. it does quoting and escaping and
/// replaces "" or "-" with "standard input".
std::string PrintableRxfilename(std::string rxfilename);

/// PrintableWxfilename turns the filename into a more human-readable
/// form for error reporting, i.e. it does quoting and escaping and
/// replaces "" or "-" with "standard output".
std::string PrintableWxfilename(std::string wxfilename);

/// @}

}  // end namespace kaldi.

#include "util/kaldi-io-inl.h"

#endif  // KALDI_UTIL_KALDI_IO_H_
