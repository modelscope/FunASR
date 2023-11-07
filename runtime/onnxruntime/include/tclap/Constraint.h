// -*- Mode: c++; c-basic-offset: 4; tab-width: 4; -*-


/******************************************************************************
 *
 *  file:  Constraint.h
 *
 *  Copyright (c) 2005, Michael E. Smoot
 *  Copyright (c) 2017, Google LLC
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

#ifndef TCLAP_CONSTRAINT_H
#define TCLAP_CONSTRAINT_H

#include <string>
#include <vector>
#include <list>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <stdexcept>

namespace TCLAP {

/**
 * The interface that defines the interaction between the Arg and Constraint.
 */
template<class T>
class Constraint
{

	public:
		/**
		 * Returns a description of the Constraint.
		 */
		virtual std::string description() const =0;

		/**
		 * Returns the short ID for the Constraint.
		 */
		virtual std::string shortID() const =0;

		/**
		 * The method used to verify that the value parsed from the command
		 * line meets the constraint.
		 * \param value - The value that will be checked.
		 */
		virtual bool check(const T& value) const =0;

		/**
		 * Destructor.
		 * Silences warnings about Constraint being a base class with virtual
		 * functions but without a virtual destructor.
		 */
		virtual ~Constraint() { ; }

		static std::string shortID(Constraint<T> *constraint) {
		  if (!constraint)
		    throw std::logic_error("Cannot create a ValueArg with a NULL constraint");
		  return constraint->shortID();
		}
};

} //namespace TCLAP
#endif
