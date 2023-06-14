///////////////////////////////////////////////////////////////////////////////
// This is a modified version of the std::basic_filebuf from libc++
// (http://libcxx.llvm.org/).
// It allows one to create basic_filebuf from an existing FILE* handle or file
// descriptor.
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source License licenses. See LICENSE.TXT for details (included at the
// bottom).
///////////////////////////////////////////////////////////////////////////////
#ifndef KALDI_UTIL_BASIC_FILEBUF_H_
#define KALDI_UTIL_BASIC_FILEBUF_H_

///////////////////////////////////////////////////////////////////////////////
#include <fstream>
#include <cstdio>
#include <cstring>
#include <string>
#include <algorithm>

///////////////////////////////////////////////////////////////////////////////
namespace kaldi {
///////////////////////////////////////////////////////////////////////////////
template <typename CharT, typename Traits = std::char_traits<CharT> >
class basic_filebuf : public std::basic_streambuf<CharT, Traits> {
 public:
    typedef CharT                            char_type;
    typedef Traits                           traits_type;
    typedef typename traits_type::int_type   int_type;
    typedef typename traits_type::pos_type   pos_type;
    typedef typename traits_type::off_type   off_type;
    typedef typename traits_type::state_type state_type;

    basic_filebuf();
    basic_filebuf(basic_filebuf&& rhs);
    virtual ~basic_filebuf();

    basic_filebuf& operator=(basic_filebuf&& rhs);
    void swap(basic_filebuf& rhs);

    bool is_open() const;
    basic_filebuf* open(const char* s, std::ios_base::openmode mode);
    basic_filebuf* open(const std::string& s, std::ios_base::openmode mode);
    basic_filebuf* open(int fd, std::ios_base::openmode mode);
    basic_filebuf* open(FILE* f, std::ios_base::openmode mode);
    basic_filebuf* close();

    FILE* file() { return this->_M_file; }
    int fd() { return fileno(this->_M_file); }

 protected:
    int_type underflow() override;
    int_type pbackfail(int_type c = traits_type::eof()) override;
    int_type overflow(int_type c = traits_type::eof()) override;
    std::basic_streambuf<char_type, traits_type>*
      setbuf(char_type* s, std::streamsize n) override;
    pos_type seekoff(off_type off, std::ios_base::seekdir way,
                     std::ios_base::openmode wch =
                     std::ios_base::in | std::ios_base::out) override;
    pos_type seekpos(pos_type sp,
                     std::ios_base::openmode wch =
                     std::ios_base::in | std::ios_base::out) override;
    int sync() override;
    void imbue(const std::locale& loc) override;

 protected:
    char*       _M_extbuf;
    const char* _M_extbufnext;
    const char* _M_extbufend;
    char _M_extbuf_min[8];
    size_t _M_ebs;
    char_type* _M_intbuf;
    size_t _M_ibs;
    FILE* _M_file;
    const std::codecvt<char_type, char, state_type>* _M_cv;
    state_type _M_st;
    state_type _M_st_last;
    std::ios_base::openmode _M_om;
    std::ios_base::openmode _M_cm;
    bool _M_owns_eb;
    bool _M_owns_ib;
    bool _M_always_noconv;

    const char* _M_get_mode(std::ios_base::openmode mode);
    bool _M_read_mode();
    void _M_write_mode();
};

///////////////////////////////////////////////////////////////////////////////
template <class CharT, class Traits>
basic_filebuf<CharT, Traits>::basic_filebuf()
    : _M_extbuf(nullptr),
      _M_extbufnext(nullptr),
      _M_extbufend(nullptr),
      _M_ebs(0),
      _M_intbuf(nullptr),
      _M_ibs(0),
      _M_file(nullptr),
      _M_cv(nullptr),
      _M_st(),
      _M_st_last(),
      _M_om(std::ios_base::openmode(0)),
      _M_cm(std::ios_base::openmode(0)),
      _M_owns_eb(false),
      _M_owns_ib(false),
      _M_always_noconv(false) {
    if (std::has_facet<std::codecvt<char_type, char, state_type> >
        (this->getloc())) {
        _M_cv = &std::use_facet<std::codecvt<char_type, char, state_type> >
          (this->getloc());
        _M_always_noconv = _M_cv->always_noconv();
    }
    setbuf(0, 4096);
}

///////////////////////////////////////////////////////////////////////////////
template <class CharT, class Traits>
basic_filebuf<CharT, Traits>::basic_filebuf(basic_filebuf&& rhs)
    : std::basic_streambuf<CharT, Traits>(rhs) {
    if (rhs._M_extbuf == rhs._M_extbuf_min) {
        _M_extbuf = _M_extbuf_min;
        _M_extbufnext = _M_extbuf + (rhs._M_extbufnext - rhs._M_extbuf);
        _M_extbufend = _M_extbuf + (rhs._M_extbufend - rhs._M_extbuf);
    } else {
        _M_extbuf = rhs._M_extbuf;
        _M_extbufnext = rhs._M_extbufnext;
        _M_extbufend = rhs._M_extbufend;
    }
    _M_ebs = rhs._M_ebs;
    _M_intbuf = rhs._M_intbuf;
    _M_ibs = rhs._M_ibs;
    _M_file = rhs._M_file;
    _M_cv = rhs._M_cv;
    _M_st = rhs._M_st;
    _M_st_last = rhs._M_st_last;
    _M_om = rhs._M_om;
    _M_cm = rhs._M_cm;
    _M_owns_eb = rhs._M_owns_eb;
    _M_owns_ib = rhs._M_owns_ib;
    _M_always_noconv = rhs._M_always_noconv;
    if (rhs.pbase()) {
        if (rhs.pbase() == rhs._M_intbuf)
            this->setp(_M_intbuf, _M_intbuf + (rhs. epptr() - rhs.pbase()));
        else
            this->setp(reinterpret_cast<char_type*>(_M_extbuf),
                       reinterpret_cast<char_type*>(_M_extbuf)
                       + (rhs. epptr() - rhs.pbase()));
        this->pbump(rhs. pptr() - rhs.pbase());
    } else if (rhs.eback()) {
        if (rhs.eback() == rhs._M_intbuf)
            this->setg(_M_intbuf, _M_intbuf + (rhs.gptr() - rhs.eback()),
                                  _M_intbuf + (rhs.egptr() - rhs.eback()));
        else
            this->setg(reinterpret_cast<char_type*>(_M_extbuf),
                       reinterpret_cast<char_type*>(_M_extbuf) +
                       (rhs.gptr() - rhs.eback()),
                       reinterpret_cast<char_type*>(_M_extbuf) +
                       (rhs.egptr() - rhs.eback()));
    }
    rhs._M_extbuf = nullptr;
    rhs._M_extbufnext = nullptr;
    rhs._M_extbufend = nullptr;
    rhs._M_ebs = 0;
    rhs._M_intbuf = nullptr;
    rhs._M_ibs = 0;
    rhs._M_file = nullptr;
    rhs._M_st = state_type();
    rhs._M_st_last = state_type();
    rhs._M_om = std::ios_base::openmode(0);
    rhs._M_cm = std::ios_base::openmode(0);
    rhs._M_owns_eb = false;
    rhs._M_owns_ib = false;
    rhs.setg(0, 0, 0);
    rhs.setp(0, 0);
}

///////////////////////////////////////////////////////////////////////////////
template <class CharT, class Traits>
inline
basic_filebuf<CharT, Traits>&
basic_filebuf<CharT, Traits>::operator=(basic_filebuf&& rhs) {
    close();
    swap(rhs);
    return *this;
}

///////////////////////////////////////////////////////////////////////////////
template <class CharT, class Traits>
basic_filebuf<CharT, Traits>::~basic_filebuf() {
    // try
    // {
    //     close();
    // }
    // catch (...)
    // {
    // }
    if (_M_owns_eb)
        delete [] _M_extbuf;
    if (_M_owns_ib)
        delete [] _M_intbuf;
}

///////////////////////////////////////////////////////////////////////////////
template <class CharT, class Traits>
void
basic_filebuf<CharT, Traits>::swap(basic_filebuf& rhs) {
    std::basic_streambuf<char_type, traits_type>::swap(rhs);
    if (_M_extbuf != _M_extbuf_min && rhs._M_extbuf != rhs._M_extbuf_min) {
        std::swap(_M_extbuf, rhs._M_extbuf);
        std::swap(_M_extbufnext, rhs._M_extbufnext);
        std::swap(_M_extbufend, rhs._M_extbufend);
    } else {
        ptrdiff_t ln = _M_extbufnext - _M_extbuf;
        ptrdiff_t le = _M_extbufend - _M_extbuf;
        ptrdiff_t rn = rhs._M_extbufnext - rhs._M_extbuf;
        ptrdiff_t re = rhs._M_extbufend - rhs._M_extbuf;
        if (_M_extbuf == _M_extbuf_min && rhs._M_extbuf != rhs._M_extbuf_min) {
            _M_extbuf = rhs._M_extbuf;
            rhs._M_extbuf = rhs._M_extbuf_min;
        } else if (_M_extbuf != _M_extbuf_min &&
            rhs._M_extbuf == rhs._M_extbuf_min) {
            rhs._M_extbuf = _M_extbuf;
            _M_extbuf = _M_extbuf_min;
        }
        _M_extbufnext = _M_extbuf + rn;
        _M_extbufend = _M_extbuf + re;
        rhs._M_extbufnext = rhs._M_extbuf + ln;
        rhs._M_extbufend = rhs._M_extbuf + le;
    }
    std::swap(_M_ebs, rhs._M_ebs);
    std::swap(_M_intbuf, rhs._M_intbuf);
    std::swap(_M_ibs, rhs._M_ibs);
    std::swap(_M_file, rhs._M_file);
    std::swap(_M_cv, rhs._M_cv);
    std::swap(_M_st, rhs._M_st);
    std::swap(_M_st_last, rhs._M_st_last);
    std::swap(_M_om, rhs._M_om);
    std::swap(_M_cm, rhs._M_cm);
    std::swap(_M_owns_eb, rhs._M_owns_eb);
    std::swap(_M_owns_ib, rhs._M_owns_ib);
    std::swap(_M_always_noconv, rhs._M_always_noconv);
    if (this->eback() == reinterpret_cast<char_type*>(rhs._M_extbuf_min)) {
        ptrdiff_t n = this->gptr() - this->eback();
        ptrdiff_t e = this->egptr() - this->eback();
        this->setg(reinterpret_cast<char_type*>(_M_extbuf_min),
                   reinterpret_cast<char_type*>(_M_extbuf_min) + n,
                   reinterpret_cast<char_type*>(_M_extbuf_min) + e);
    } else if (this->pbase() ==
               reinterpret_cast<char_type*>(rhs._M_extbuf_min)) {
        ptrdiff_t n = this->pptr() - this->pbase();
        ptrdiff_t e = this->epptr() - this->pbase();
        this->setp(reinterpret_cast<char_type*>(_M_extbuf_min),
                   reinterpret_cast<char_type*>(_M_extbuf_min) + e);
        this->pbump(n);
    }
    if (rhs.eback() == reinterpret_cast<char_type*>(_M_extbuf_min)) {
        ptrdiff_t n = rhs.gptr() - rhs.eback();
        ptrdiff_t e = rhs.egptr() - rhs.eback();
        rhs.setg(reinterpret_cast<char_type*>(rhs._M_extbuf_min),
                 reinterpret_cast<char_type*>(rhs._M_extbuf_min) + n,
                 reinterpret_cast<char_type*>(rhs._M_extbuf_min) + e);
    } else if (rhs.pbase() == reinterpret_cast<char_type*>(_M_extbuf_min)) {
        ptrdiff_t n = rhs.pptr() - rhs.pbase();
        ptrdiff_t e = rhs.epptr() - rhs.pbase();
        rhs.setp(reinterpret_cast<char_type*>(rhs._M_extbuf_min),
                 reinterpret_cast<char_type*>(rhs._M_extbuf_min) + e);
        rhs.pbump(n);
    }
}

///////////////////////////////////////////////////////////////////////////////
template <class CharT, class Traits>
inline
void
swap(basic_filebuf<CharT, Traits>& x, basic_filebuf<CharT, Traits>& y) {
    x.swap(y);
}

///////////////////////////////////////////////////////////////////////////////
template <class CharT, class Traits>
inline
bool
basic_filebuf<CharT, Traits>::is_open() const {
    return _M_file != nullptr;
}

///////////////////////////////////////////////////////////////////////////////
template <class CharT, class Traits>
const char* basic_filebuf<CharT, Traits>::
_M_get_mode(std::ios_base::openmode mode) {
    switch ((mode & ~std::ios_base::ate) | 0) {
    case std::ios_base::out:
    case std::ios_base::out | std::ios_base::trunc:
        return "w";
    case std::ios_base::out | std::ios_base::app:
    case std::ios_base::app:
        return "a";
        break;
    case std::ios_base::in:
        return "r";
    case std::ios_base::in  | std::ios_base::out:
        return "r+";
    case std::ios_base::in  | std::ios_base::out | std::ios_base::trunc:
        return "w+";
    case std::ios_base::in  | std::ios_base::out | std::ios_base::app:
    case std::ios_base::in  | std::ios_base::app:
        return "a+";
    case std::ios_base::out | std::ios_base::binary:
    case std::ios_base::out | std::ios_base::trunc | std::ios_base::binary:
        return "wb";
    case std::ios_base::out | std::ios_base::app | std::ios_base::binary:
    case std::ios_base::app | std::ios_base::binary:
        return "ab";
    case std::ios_base::in  | std::ios_base::binary:
        return "rb";
    case std::ios_base::in  | std::ios_base::out | std::ios_base::binary:
        return "r+b";
    case std::ios_base::in  | std::ios_base::out | std::ios_base::trunc |
      std::ios_base::binary:
        return "w+b";
    case std::ios_base::in  | std::ios_base::out | std::ios_base::app |
      std::ios_base::binary:
    case std::ios_base::in  | std::ios_base::app | std::ios_base::binary:
        return "a+b";
    default:
        return nullptr;
    }
}

///////////////////////////////////////////////////////////////////////////////
template <class CharT, class Traits>
basic_filebuf<CharT, Traits>*
basic_filebuf<CharT, Traits>::
open(const char* s, std::ios_base::openmode mode) {
    basic_filebuf<CharT, Traits>* rt = nullptr;
    if (_M_file == nullptr) {
        const char* md= _M_get_mode(mode);
        if (md) {
            _M_file = fopen(s, md);
            if (_M_file) {
                rt = this;
                _M_om = mode;
                if (mode & std::ios_base::ate) {
                    if (fseek(_M_file, 0, SEEK_END)) {
                        fclose(_M_file);
                        _M_file = nullptr;
                        rt = nullptr;
                    }
                }
            }
        }
    }
    return rt;
}

///////////////////////////////////////////////////////////////////////////////
template <class CharT, class Traits>
inline
basic_filebuf<CharT, Traits>*
basic_filebuf<CharT, Traits>::open(const std::string& s,
    std::ios_base::openmode mode) {
    return open(s.c_str(), mode);
}

///////////////////////////////////////////////////////////////////////////////
template <class CharT, class Traits>
basic_filebuf<CharT, Traits>*
basic_filebuf<CharT, Traits>::open(int fd, std::ios_base::openmode mode) {
    const char* md= this->_M_get_mode(mode);
    if (md) {
        this->_M_file= fdopen(fd, md);
        this->_M_om = mode;
        return this;
    } else {
      return nullptr;
    }
}

///////////////////////////////////////////////////////////////////////////////
template <class CharT, class Traits>
basic_filebuf<CharT, Traits>*
basic_filebuf<CharT, Traits>::open(FILE* f, std::ios_base::openmode mode) {
    this->_M_file = f;
    this->_M_om = mode;
    return this;
}

///////////////////////////////////////////////////////////////////////////////
template <class CharT, class Traits>
basic_filebuf<CharT, Traits>*
basic_filebuf<CharT, Traits>::close() {
    basic_filebuf<CharT, Traits>* rt = nullptr;
    if (_M_file) {
        rt = this;
        std::unique_ptr<FILE, int(*)(FILE*)> h(_M_file, fclose);
        if (sync())
            rt = nullptr;
        if (fclose(h.release()) == 0)
            _M_file = nullptr;
        else
            rt = nullptr;
    }
    return rt;
}

///////////////////////////////////////////////////////////////////////////////
template <class CharT, class Traits>
typename basic_filebuf<CharT, Traits>::int_type
basic_filebuf<CharT, Traits>::underflow() {
    if (_M_file == nullptr)
        return traits_type::eof();
    bool initial = _M_read_mode();
    char_type buf;
    if (this->gptr() == nullptr)
        this->setg(&buf, &buf+1, &buf+1);
    const size_t unget_sz = initial ? 0 : std::
      min<size_t>((this->egptr() - this->eback()) / 2, 4);
    int_type c = traits_type::eof();
    if (this->gptr() == this->egptr()) {
        memmove(this->eback(), this->egptr() - unget_sz,
            unget_sz * sizeof(char_type));
        if (_M_always_noconv) {
            size_t nmemb = static_cast<size_t>
              (this->egptr() - this->eback() - unget_sz);
            nmemb = fread(this->eback() + unget_sz, 1, nmemb, _M_file);
            if (nmemb != 0) {
                this->setg(this->eback(),
                           this->eback() + unget_sz,
                           this->eback() + unget_sz + nmemb);
                c = traits_type::to_int_type(*this->gptr());
            }
        } else {
            memmove(_M_extbuf, _M_extbufnext, _M_extbufend - _M_extbufnext);
            _M_extbufnext = _M_extbuf + (_M_extbufend - _M_extbufnext);
            _M_extbufend = _M_extbuf +
              (_M_extbuf == _M_extbuf_min ? sizeof(_M_extbuf_min) : _M_ebs);
            size_t nmemb = std::min(static_cast<size_t>(_M_ibs - unget_sz),
                                    static_cast<size_t>
                                    (_M_extbufend - _M_extbufnext));
            std::codecvt_base::result r;
            _M_st_last = _M_st;
            size_t nr = fread(
                reinterpret_cast<void*>(const_cast<char_type*>(_M_extbufnext)),
                1, nmemb, _M_file);
            if (nr != 0) {
                if (!_M_cv)
                    throw std::bad_cast();
                _M_extbufend = _M_extbufnext + nr;
                char_type*  inext;
                r = _M_cv->in(_M_st, _M_extbuf, _M_extbufend, _M_extbufnext,
                              this->eback() + unget_sz,
                              this->eback() + _M_ibs, inext);
                if (r == std::codecvt_base::noconv) {
                    this->setg(reinterpret_cast<char_type*>(_M_extbuf),
                               reinterpret_cast<char_type*>(_M_extbuf),
                               const_cast<char_type*>(_M_extbufend));
                    c = traits_type::to_int_type(*this->gptr());
                } else if (inext != this->eback() + unget_sz) {
                    this->setg(this->eback(), this->eback() + unget_sz, inext);
                    c = traits_type::to_int_type(*this->gptr());
                }
            }
        }
    } else {
        c = traits_type::to_int_type(*this->gptr());
    }
    if (this->eback() == &buf)
        this->setg(0, 0, 0);
    return c;
}

///////////////////////////////////////////////////////////////////////////////
template <class CharT, class Traits>
typename basic_filebuf<CharT, Traits>::int_type
basic_filebuf<CharT, Traits>::pbackfail(int_type c) {
    if (_M_file && this->eback() < this->gptr()) {
        if (traits_type::eq_int_type(c, traits_type::eof())) {
            this->gbump(-1);
            return traits_type::not_eof(c);
        }
        if ((_M_om & std::ios_base::out) ||
            traits_type::eq(traits_type::to_char_type(c), this->gptr()[-1])) {
            this->gbump(-1);
            *this->gptr() = traits_type::to_char_type(c);
            return c;
        }
    }
    return traits_type::eof();
}

///////////////////////////////////////////////////////////////////////////////
template <class CharT, class Traits>
typename basic_filebuf<CharT, Traits>::int_type
basic_filebuf<CharT, Traits>::overflow(int_type c) {
    if (_M_file == nullptr)
        return traits_type::eof();
    _M_write_mode();
    char_type buf;
    char_type* pb_save = this->pbase();
    char_type* epb_save = this->epptr();
    if (!traits_type::eq_int_type(c, traits_type::eof())) {
        if (this->pptr() == nullptr)
            this->setp(&buf, &buf+1);
        *this->pptr() = traits_type::to_char_type(c);
        this->pbump(1);
    }
    if (this->pptr() != this->pbase()) {
        if (_M_always_noconv) {
            size_t nmemb = static_cast<size_t>(this->pptr() - this->pbase());
            if (fwrite(this->pbase(), sizeof(char_type),
                  nmemb, _M_file) != nmemb)
                return traits_type::eof();
        } else {
            char* extbe = _M_extbuf;
            std::codecvt_base::result r;
            do {
                if (!_M_cv)
                    throw std::bad_cast();
                const char_type* e;
                r = _M_cv->out(_M_st, this->pbase(), this->pptr(), e,
                               _M_extbuf, _M_extbuf + _M_ebs, extbe);
                if (e == this->pbase())
                    return traits_type::eof();
                if (r == std::codecvt_base::noconv) {
                    size_t nmemb = static_cast<size_t>
                      (this->pptr() - this->pbase());
                    if (fwrite(this->pbase(), 1, nmemb, _M_file) != nmemb)
                        return traits_type::eof();
                } else if (r == std::codecvt_base::ok ||
                    r == std::codecvt_base::partial) {
                    size_t nmemb = static_cast<size_t>(extbe - _M_extbuf);
                    if (fwrite(_M_extbuf, 1, nmemb, _M_file) != nmemb)
                        return traits_type::eof();
                    if (r == std::codecvt_base::partial) {
                        this->setp(const_cast<char_type*>(e),
                            this->pptr());
                        this->pbump(this->epptr() - this->pbase());
                    }
                } else {
                    return traits_type::eof();
                }
            } while (r == std::codecvt_base::partial);
        }
        this->setp(pb_save, epb_save);
    }
    return traits_type::not_eof(c);
}

///////////////////////////////////////////////////////////////////////////////
template <class CharT, class Traits>
std::basic_streambuf<CharT, Traits>*
basic_filebuf<CharT, Traits>::setbuf(char_type* s, std::streamsize n) {
    this->setg(0, 0, 0);
    this->setp(0, 0);
    if (_M_owns_eb)
        delete [] _M_extbuf;
    if (_M_owns_ib)
        delete [] _M_intbuf;
    _M_ebs = n;
    if (_M_ebs > sizeof(_M_extbuf_min)) {
        if (_M_always_noconv && s) {
            _M_extbuf = reinterpret_cast<char*>(s);
            _M_owns_eb = false;
        } else {
            _M_extbuf = new char[_M_ebs];
            _M_owns_eb = true;
        }
    } else {
        _M_extbuf = _M_extbuf_min;
        _M_ebs = sizeof(_M_extbuf_min);
        _M_owns_eb = false;
    }
    if (!_M_always_noconv) {
        _M_ibs = std::max<std::streamsize>(n, sizeof(_M_extbuf_min));
        if (s && _M_ibs >= sizeof(_M_extbuf_min)) {
            _M_intbuf = s;
            _M_owns_ib = false;
        } else {
            _M_intbuf = new char_type[_M_ibs];
            _M_owns_ib = true;
        }
    } else {
        _M_ibs = 0;
        _M_intbuf = 0;
        _M_owns_ib = false;
    }
    return this;
}

///////////////////////////////////////////////////////////////////////////////
template <class CharT, class Traits>
typename basic_filebuf<CharT, Traits>::pos_type
basic_filebuf<CharT, Traits>::seekoff(off_type off, std::ios_base::seekdir way,
                                      std::ios_base::openmode) {
    if (!_M_cv)
        throw std::bad_cast();
    int width = _M_cv->encoding();
    if (_M_file == nullptr || (width <= 0 && off != 0) || sync())
        return pos_type(off_type(-1));
    // width > 0 || off == 0
    int whence;
    switch (way) {
    case std::ios_base::beg:
        whence = SEEK_SET;
        break;
    case std::ios_base::cur:
        whence = SEEK_CUR;
        break;
    case std::ios_base::end:
        whence = SEEK_END;
        break;
    default:
        return pos_type(off_type(-1));
    }
#if _WIN32
    if (fseek(_M_file, width > 0 ? width * off : 0, whence))
        return pos_type(off_type(-1));
    pos_type r = ftell(_M_file);
#else
    if (fseeko(_M_file, width > 0 ? width * off : 0, whence))
        return pos_type(off_type(-1));
    pos_type r = ftello(_M_file);
#endif
    r.state(_M_st);
    return r;
}

///////////////////////////////////////////////////////////////////////////////
template <class CharT, class Traits>
typename basic_filebuf<CharT, Traits>::pos_type
basic_filebuf<CharT, Traits>::seekpos(pos_type sp, std::ios_base::openmode) {
    if (_M_file == nullptr || sync())
        return pos_type(off_type(-1));
#if _WIN32
    if (fseek(_M_file, sp, SEEK_SET))
        return pos_type(off_type(-1));
#else
    if (fseeko(_M_file, sp, SEEK_SET))
        return pos_type(off_type(-1));
#endif
    _M_st = sp.state();
    return sp;
}

///////////////////////////////////////////////////////////////////////////////
template <class CharT, class Traits>
int
basic_filebuf<CharT, Traits>::sync() {
    if (_M_file == nullptr)
        return 0;
    if (!_M_cv)
        throw std::bad_cast();
    if (_M_cm & std::ios_base::out) {
        if (this->pptr() != this->pbase())
            if (overflow() == traits_type::eof())
                return -1;
        std::codecvt_base::result r;
        do {
            char* extbe;
            r = _M_cv->unshift(_M_st, _M_extbuf, _M_extbuf + _M_ebs, extbe);
            size_t nmemb = static_cast<size_t>(extbe - _M_extbuf);
            if (fwrite(_M_extbuf, 1, nmemb, _M_file) != nmemb)
                return -1;
        } while (r == std::codecvt_base::partial);
        if (r == std::codecvt_base::error)
            return -1;
        if (fflush(_M_file))
            return -1;
    } else if (_M_cm & std::ios_base::in) {
        off_type c;
        state_type state = _M_st_last;
        bool update_st = false;
        if (_M_always_noconv) {
            c = this->egptr() - this->gptr();
        } else {
            int width = _M_cv->encoding();
            c = _M_extbufend - _M_extbufnext;
            if (width > 0) {
                c += width * (this->egptr() - this->gptr());
            } else {
                if (this->gptr() != this->egptr()) {
                    const int off = _M_cv->length(state, _M_extbuf,
                                                  _M_extbufnext,
                                                  this->gptr() - this->eback());
                    c += _M_extbufnext - _M_extbuf - off;
                    update_st = true;
                }
            }
        }
#if _WIN32
        if (fseek(_M_file_, -c, SEEK_CUR))
            return -1;
#else
        if (fseeko(_M_file, -c, SEEK_CUR))
            return -1;
#endif
        if (update_st)
            _M_st = state;
        _M_extbufnext = _M_extbufend = _M_extbuf;
        this->setg(0, 0, 0);
        _M_cm = std::ios_base::openmode(0);
    }
    return 0;
}

///////////////////////////////////////////////////////////////////////////////
template <class CharT, class Traits>
void
basic_filebuf<CharT, Traits>::imbue(const std::locale& loc) {
    sync();
    _M_cv = &std::use_facet<std::codecvt<char_type, char, state_type> >(loc);
    bool old_anc = _M_always_noconv;
    _M_always_noconv = _M_cv->always_noconv();
    if (old_anc != _M_always_noconv) {
        this->setg(0, 0, 0);
        this->setp(0, 0);
        // invariant, char_type is char, else we couldn't get here
        // need to dump _M_intbuf
        if (_M_always_noconv) {
            if (_M_owns_eb)
                delete [] _M_extbuf;
            _M_owns_eb = _M_owns_ib;
            _M_ebs = _M_ibs;
            _M_extbuf = reinterpret_cast<char*>(_M_intbuf);
            _M_ibs = 0;
            _M_intbuf = nullptr;
            _M_owns_ib = false;
        } else {  // need to obtain an _M_intbuf.
         // If _M_extbuf is user-supplied, use it, else new _M_intbuf
            if (!_M_owns_eb && _M_extbuf != _M_extbuf_min) {
                _M_ibs = _M_ebs;
                _M_intbuf = reinterpret_cast<char_type*>(_M_extbuf);
                _M_owns_ib = false;
                _M_extbuf = new char[_M_ebs];
                _M_owns_eb = true;
            } else {
                _M_ibs = _M_ebs;
                _M_intbuf = new char_type[_M_ibs];
                _M_owns_ib = true;
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
template <class CharT, class Traits>
bool
basic_filebuf<CharT, Traits>::_M_read_mode() {
    if (!(_M_cm & std::ios_base::in)) {
        this->setp(0, 0);
        if (_M_always_noconv)
            this->setg(reinterpret_cast<char_type*>(_M_extbuf),
                       reinterpret_cast<char_type*>(_M_extbuf) + _M_ebs,
                       reinterpret_cast<char_type*>(_M_extbuf) + _M_ebs);
        else
            this->setg(_M_intbuf, _M_intbuf + _M_ibs, _M_intbuf + _M_ibs);
        _M_cm = std::ios_base::in;
        return true;
    }
    return false;
}

///////////////////////////////////////////////////////////////////////////////
template <class CharT, class Traits>
void
basic_filebuf<CharT, Traits>::_M_write_mode() {
    if (!(_M_cm & std::ios_base::out)) {
        this->setg(0, 0, 0);
        if (_M_ebs > sizeof(_M_extbuf_min)) {
            if (_M_always_noconv)
                this->setp(reinterpret_cast<char_type*>(_M_extbuf),
                           reinterpret_cast<char_type*>(_M_extbuf) +
                           (_M_ebs - 1));
            else
                this->setp(_M_intbuf, _M_intbuf + (_M_ibs - 1));
        } else {
            this->setp(0, 0);
        }
        _M_cm = std::ios_base::out;
    }
}

///////////////////////////////////////////////////////////////////////////////
}

///////////////////////////////////////////////////////////////////////////////
#endif  // KALDI_UTIL_BASIC_FILEBUF_H_

///////////////////////////////////////////////////////////////////////////////

/*
 * ============================================================================
 * libc++ License
 * ============================================================================
 *
 * The libc++ library is dual licensed under both the University of Illinois
 * "BSD-Like" license and the MIT license.  As a user of this code you may
 * choose to use it under either license.  As a contributor, you agree to allow
 * your code to be used under both.
 *
 * Full text of the relevant licenses is included below.
 *
 * ============================================================================
 *
 * University of Illinois/NCSA
 * Open Source License
 *
 * Copyright (c) 2009-2014 by the contributors listed in CREDITS.TXT (included below)
 *
 * All rights reserved.
 *
 * Developed by:
 *
 *     LLVM Team
 *
 *     University of Illinois at Urbana-Champaign
 *
 *     http://llvm.org
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal with
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is furnished to do
 * so, subject to the following conditions:
 *
 *     * Redistributions of source code must retain the above copyright notice,
 *       this list of conditions and the following disclaimers.
 *
 *     * Redistributions in binary form must reproduce the above copyright notice,
 *       this list of conditions and the following disclaimers in the
 *       documentation and/or other materials provided with the distribution.
 *
 *     * Neither the names of the LLVM Team, University of Illinois at
 *       Urbana-Champaign, nor the names of its contributors may be used to
 *       endorse or promote products derived from this Software without specific
 *       prior written permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH THE
 * SOFTWARE.
 *
 * ==============================================================================
 *
 * Copyright (c) 2009-2014 by the contributors listed in CREDITS.TXT (included below)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ==============================================================================
 *
 * This file is a partial list of people who have contributed to the LLVM/libc++
 * project.  If you have contributed a patch or made some other contribution to
 * LLVM/libc++, please submit a patch to this file to add yourself, and it will be
 * done!
 *
 * The list is sorted by surname and formatted to allow easy grepping and
 * beautification by scripts.  The fields are: name (N), email (E), web-address
 * (W), PGP key ID and fingerprint (P), description (D), and snail-mail address
 * (S).
 *
 * N: Saleem Abdulrasool
 * E: compnerd@compnerd.org
 * D: Minor patches and Linux fixes.
 *
 * N: Dimitry Andric
 * E: dimitry@andric.com
 * D: Visibility fixes, minor FreeBSD portability patches.
 *
 * N: Holger Arnold
 * E: holgerar@gmail.com
 * D: Minor fix.
 *
 * N: Ruben Van Boxem
 * E: vanboxem dot ruben at gmail dot com
 * D: Initial Windows patches.
 *
 * N: David Chisnall
 * E: theraven at theravensnest dot org
 * D: FreeBSD and Solaris ports, libcxxrt support, some atomics work.
 *
 * N: Marshall Clow
 * E: mclow.lists@gmail.com
 * E: marshall@idio.com
 * D: C++14 support, patches and bug fixes.
 *
 * N: Bill Fisher
 * E: william.w.fisher@gmail.com
 * D: Regex bug fixes.
 *
 * N: Matthew Dempsky
 * E: matthew@dempsky.org
 * D: Minor patches and bug fixes.
 *
 * N: Google Inc.
 * D: Copyright owner and contributor of the CityHash algorithm
 *
 * N: Howard Hinnant
 * E: hhinnant@apple.com
 * D: Architect and primary author of libc++
 *
 * N: Hyeon-bin Jeong
 * E: tuhertz@gmail.com
 * D: Minor patches and bug fixes.
 *
 * N: Argyrios Kyrtzidis
 * E: kyrtzidis@apple.com
 * D: Bug fixes.
 *
 * N: Bruce Mitchener, Jr.
 * E: bruce.mitchener@gmail.com
 * D: Emscripten-related changes.
 *
 * N: Michel Morin
 * E: mimomorin@gmail.com
 * D: Minor patches to is_convertible.
 *
 * N: Andrew Morrow
 * E: andrew.c.morrow@gmail.com
 * D: Minor patches and Linux fixes.
 *
 * N: Arvid Picciani
 * E: aep at exys dot org
 * D: Minor patches and musl port.
 *
 * N: Bjorn Reese
 * E: breese@users.sourceforge.net
 * D: Initial regex prototype
 *
 * N: Nico Rieck
 * E: nico.rieck@gmail.com
 * D: Windows fixes
 *
 * N: Jonathan Sauer
 * D: Minor patches, mostly related to constexpr
 *
 * N: Craig Silverstein
 * E: csilvers@google.com
 * D: Implemented Cityhash as the string hash function on 64-bit machines
 *
 * N: Richard Smith
 * D: Minor patches.
 *
 * N: Joerg Sonnenberger
 * E: joerg@NetBSD.org
 * D: NetBSD port.
 *
 * N: Stephan Tolksdorf
 * E: st@quanttec.com
 * D: Minor <atomic> fix
 *
 * N: Michael van der Westhuizen
 * E: r1mikey at gmail dot com
 *
 * N: Klaas de Vries
 * E: klaas at klaasgaaf dot nl
 * D: Minor bug fix.
 *
 * N: Zhang Xiongpang
 * E: zhangxiongpang@gmail.com
 * D: Minor patches and bug fixes.
 *
 * N: Xing Xue
 * E: xingxue@ca.ibm.com
 * D: AIX port
 *
 * N: Zhihao Yuan
 * E: lichray@gmail.com
 * D: Standard compatibility fixes.
 *
 * N: Jeffrey Yasskin
 * E: jyasskin@gmail.com
 * E: jyasskin@google.com
 * D: Linux fixes.
 */
