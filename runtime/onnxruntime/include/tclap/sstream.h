// -*- Mode: c++; c-basic-offset: 4; tab-width: 4; -*-

/******************************************************************************
 *
 *  file:  sstream.h
 *
 *  Copyright (c) 2003, Michael E. Smoot .
 *  Copyright (c) 2004, Michael E. Smoot, Daniel Aarno .
 *  Copyright (c) 2017 Google Inc.
 *  All rights reserved.
 *
 *  See the file COPYING in the top directory of this distribution for
 *  more information.
 *
 *  THE SOFTWARE IS PROVIDED _AS IS_, WITHOUT WARRANTY OF ANY KIND, EXPRESS
 *  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 *  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 *  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 *  DEALINGS IN THE SOFTWARE.
 *
 *****************************************************************************/

#ifndef TCLAP_SSTREAM_H
#define TCLAP_SSTREAM_H

#if !defined(HAVE_STRSTREAM)
// Assume sstream is available if strstream is not specified
// (https://sourceforge.net/p/tclap/bugs/23/)
#define HAVE_SSTREAM
#endif

#if defined(HAVE_SSTREAM)
#include <sstream>
namespace TCLAP {
    typedef std::istringstream istringstream;
    typedef std::ostringstream ostringstream;
}
#elif defined(HAVE_STRSTREAM)
#include <strstream>
namespace TCLAP {
    typedef std::istrstream istringstream;
    typedef std::ostrstream ostringstream;
}
#else
#error "Need a stringstream (sstream or strstream) to compile!"
#endif

#endif  // TCLAP_SSTREAM_H
