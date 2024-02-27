#pragma once

#include <cstdint>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

template <typename T> class CircularQueue
{
  public:
    explicit CircularQueue(std::size_t capacity) : buffer_(capacity), head_(0), tail_(0), count_(0)
    {
    }

    ~CircularQueue() noexcept
    {
        clear();
    }

    inline bool empty() const noexcept
    {
        return (count_ == 0);
    }

    inline bool full() const noexcept
    {
        return (count_ == buffer_.size());
    }

    inline std::size_t size() const noexcept
    {
        return count_;
    }

    inline void push(const T &value) noexcept
    {
        emplace(value);
    }

    inline void push(T &&value) noexcept
    {
        emplace(std::move(value));
    }

    template <typename... Args> inline void emplace(Args &&...args) noexcept
    {
        ::new (&buffer_[tail_]) T{std::forward<Args>(args)...};

        ++tail_;
        if (tail_ >= buffer_.size())
        {
            tail_ = 0;
        }

        ++count_;
    }

    inline void pop() noexcept
    {
        reinterpret_cast<T *>(&buffer_[head_])->~T();

        ++head_;
        if (head_ >= buffer_.size())
        {
            head_ = 0;
        }

        --count_;
    }

    inline T &front() noexcept
    {
        return *reinterpret_cast<T *>(&buffer_[head_]);
    }

    inline const T &front() const noexcept
    {
        return *reinterpret_cast<const T *>(&buffer_[head_]);
    }

    inline T &back() noexcept
    {
        const std::size_t back_index = (tail_ == 0) ? (buffer_.size() - 1) : (tail_ - 1);
        return *reinterpret_cast<T *>(&buffer_[back_index]);
    }

    inline const T &back() const noexcept
    {
        const std::size_t back_index = (tail_ == 0) ? (buffer_.size() - 1) : (tail_ - 1);
        return *reinterpret_cast<const T *>(&buffer_[back_index]);
    }

  private:
    std::vector<std::aligned_storage_t<sizeof(T), alignof(T)>> buffer_;
    std::size_t head_;
    std::size_t tail_;
    std::size_t count_;

    inline void clear() noexcept
    {
        while (!empty())
        {
            pop();
        }
    }
};
