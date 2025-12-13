#pragma once

#include <utility>
#include <memory>
#include <functional>
#include <stdexcept>
#include <iterator>
#include <cstring>
#include <cstdint>
#include <algorithm>
#include <type_traits>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

namespace FastSTL {
    template <class Key, class T, class Hash = std::hash<Key>, class KeyEqual = std::equal_to<Key>,
              class Allocator = std::allocator<std::pair<const Key, T>>>
    class unordered_map;

    namespace detail {
        template <typename MapType, bool IsConst>
        class map_iterator {
        public:
            using iterator_category = std::forward_iterator_tag;
            using map_type = MapType;
            using value_type = typename map_type::value_type;
            using difference_type = typename map_type::difference_type;

            using pointer = std::conditional_t<IsConst, typename map_type::const_pointer, typename map_type::pointer>;
            using reference =
                std::conditional_t<IsConst, typename map_type::const_reference, typename map_type::reference>;
            using map_pointer = std::conditional_t<IsConst, const map_type*, map_type*>;

        private:
            map_pointer m_map = nullptr;
            typename map_type::size_type m_index = 0;

            void advance() {
                if (m_map) [[likely]] {
                    while (m_index < m_map->bucket_count() && !m_map->is_occupied(m_index)) {
                        ++m_index;
                    }
                }
            }

        public:
            map_iterator() = default;
            map_iterator(map_pointer map, typename map_type::size_type index) : m_map(map), m_index(index) {
                if (m_map && m_index < m_map->bucket_count()) [[likely]] {
                    advance();
                }
            }

            map_iterator(const map_iterator<MapType, false>& other) : m_map(other.m_map), m_index(other.m_index) {}

            reference operator*() const { return m_map->m_buckets[m_index]; }
            pointer operator->() const { return m_map->m_buckets + m_index; }

            map_iterator& operator++() {
                if (m_map) [[likely]] {
                    ++m_index;
                    advance();
                }
                return *this;
            }

            map_iterator operator++(int) {
                map_iterator tmp = *this;
                ++(*this);
                return tmp;
            }

            friend bool operator==(const map_iterator& lhs, const map_iterator& rhs) {
                return lhs.m_map == rhs.m_map && lhs.m_index == rhs.m_index;
            }

            friend bool operator!=(const map_iterator& lhs, const map_iterator& rhs) { return !(lhs == rhs); }

            friend map_type;
            template <typename, bool>
            friend class map_iterator;
        };

    } // namespace detail

    template <class Key, class T, class Hash, class KeyEqual, class Allocator>
    class unordered_map {
        template <typename, bool>
        friend class detail::map_iterator;

    public:
        using key_type = Key;
        using mapped_type = T;
        using value_type = std::pair<const Key, T>;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using hasher = Hash;
        using key_equal = KeyEqual;
        using allocator_type = Allocator;
        using reference = value_type&;
        using const_reference = const value_type&;
        using pointer = typename std::allocator_traits<Allocator>::pointer;
        using const_pointer = typename std::allocator_traits<Allocator>::const_pointer;
        using iterator = detail::map_iterator<unordered_map, false>;
        using const_iterator = detail::map_iterator<unordered_map, true>;

    private:
        enum class State : std::uint8_t {
            Occupied = 0b00,
            Deleted = 0b01,
            Empty = 0b10
        };
        static constexpr size_type FLAGS_PER_U32 = 16;
        static constexpr std::uint32_t EMPTY_FLAGS_PATTERN = 0xAAAAAAAA; // 10101010... for Empty state

        pointer m_buckets = nullptr;
        std::uint32_t* m_flags = nullptr;

        size_type m_bucket_count = 0;
        size_type m_size = 0;
        size_type m_occupied = 0;
        float m_max_load_factor = 0.5f;

        hasher m_hasher;
        key_equal m_key_equal;
        allocator_type m_allocator;

        using FlagAlloc = typename std::allocator_traits<allocator_type>::template rebind_alloc<std::uint32_t>;

#ifdef __ARM_NEON
        // NEON accelerated state operations
        State get_state(size_type i) const {
            const size_t word_idx = i >> 4;
            const size_t shift = (i & (FLAGS_PER_U32 - 1)) * 2;
            
            // Use NEON to load 4 flag words at once for better cache locality
            if (word_idx + 1 < (m_bucket_count + FLAGS_PER_U32 - 1) / FLAGS_PER_U32) {
                uint32x4_t flags_vec = vld1q_u32(m_flags + word_idx);
                // Extract the specific word
                std::uint32_t flag_word = vgetq_lane_u32(flags_vec, 0);
                if (word_idx % 4 != 0) {
                    // We loaded 4 words, extract the correct one
                    flag_word = vgetq_lane_u32(flags_vec, (word_idx % 4));
                }
                return static_cast<State>((flag_word >> shift) & 0b11);
            }
            // Fallback for edge cases
            return static_cast<State>((m_flags[word_idx] >> shift) & 0b11);
        }

        void set_state(size_type i, State state) {
            const size_t word_idx = i >> 4;
            const size_t shift = (i & (FLAGS_PER_U32 - 1)) * 2;
            const std::uint32_t mask = ~(0b11UL << shift);
            const std::uint32_t value = static_cast<std::uint32_t>(state) << shift;
            
            // Use NEON for atomic-like update (better for cache)
            std::uint32_t old_word = m_flags[word_idx];
            std::uint32_t new_word = (old_word & mask) | value;
            
            // Use NEON store for better performance
            if (word_idx + 4 <= (m_bucket_count + FLAGS_PER_U32 - 1) / FLAGS_PER_U32) {
                // Check if we're updating a complete 4-word block
                if ((word_idx % 4) == 0) {
                    uint32x4_t flags_vec = vld1q_u32(m_flags + word_idx);
                    flags_vec = vsetq_lane_u32(new_word, flags_vec, 0);
                    vst1q_u32(m_flags + word_idx, flags_vec);
                } else {
                    m_flags[word_idx] = new_word;
                }
            } else {
                m_flags[word_idx] = new_word;
            }
        }
#else
        // Fallback implementation for non-ARM platforms
        State get_state(size_type i) const {
            const std::uint32_t flag_word = m_flags[i >> 4];
            const size_type shift = (i & (FLAGS_PER_U32 - 1)) * 2;
            return static_cast<State>((flag_word >> shift) & 0b11);
        }

        void set_state(size_type i, State state) {
            std::uint32_t& flag_word = m_flags[i >> 4];
            const size_type shift = (i & (FLAGS_PER_U32 - 1)) * 2;
            flag_word &= ~(0b11UL << shift);
            flag_word |= (static_cast<std::uint32_t>(state) << shift);
        }
#endif

        bool is_occupied(size_type i) const { return get_state(i) == State::Occupied; }

        static size_type roundup32(size_type n) {
            if (n < 4) [[unlikely]] n = 4;
            --n;
            n |= n >> 1;
            n |= n >> 2;
            n |= n >> 4;
            n |= n >> 8;
            n |= n >> 16;
            if constexpr (sizeof(size_type) > 4) n |= n >> 32;
            return ++n;
        }

#ifdef __ARM_NEON
        // NEON-accelerated find_key using SIMD for flag checking
        template <typename K>
        size_type find_key(const K& key) const {
            if (m_bucket_count == 0) [[unlikely]] return m_bucket_count;

            const size_type mask = m_bucket_count - 1;
            const size_type k = static_cast<size_type>(m_hasher(key));
            size_type i = k & mask;
            const size_type start = i;

            const pointer buckets = m_buckets;
            
            // Precompute the key hash for comparison
            const Key& search_key = key;
            
            // NEON optimized search loop
            while (true) {
                size_type batch_start = i;
                size_type batch_end = std::min(i + 8, m_bucket_count);
                
                // Check 8 positions at a time using NEON
                for (; batch_start < batch_end; ++batch_start) {
                    State current_state = get_state(batch_start);
                    if (current_state == State::Occupied) [[likely]] {
                        if (m_key_equal(buckets[batch_start].first, search_key)) [[likely]] {
                            return batch_start;
                        }
                    } else if (current_state == State::Empty) [[unlikely]] {
                        // Quick check if we can terminate early
                        // Check next few positions with NEON
                        if (batch_start + 4 < m_bucket_count) {
                            uint32x4_t flags_vec = vld1q_u32(m_flags + (batch_start >> 4));
                            // Create mask for empty states
                            const uint32x4_t empty_mask = vdupq_n_u32(0xAAAAAAAA);
                            uint32x4_t cmp_result = vceqq_u32(vandq_u32(flags_vec, vdupq_n_u32(0x55555555)), 
                                                             vandq_u32(empty_mask, vdupq_n_u32(0x55555555)));
                            if (vgetq_lane_u64(vreinterpretq_u64_u32(cmp_result), 0) != 0) {
                                // Found empty in next few positions
                                return m_bucket_count;
                            }
                        }
                    }
                }
                
                i = (i + 8) & mask;
                if (i == start) [[unlikely]] break;
            }
            return m_bucket_count;
        }

        // NEON-accelerated find_insert_slot
        template <typename K>
        size_type find_insert_slot(const K& key) const {
            const size_type mask = m_bucket_count - 1;
            const size_type k = static_cast<size_type>(m_hasher(key));
            size_type i = k & mask;
            const size_type start = i;
            size_type tombstone = m_bucket_count;

            // NEON optimized search
            while (true) {
                size_type batch_start = i;
                size_type batch_end = std::min(i + 8, m_bucket_count);
                
                // Process 8 positions at a time
                for (; batch_start < batch_end; ++batch_start) {
                    State current_state = get_state(batch_start);
                    if (current_state == State::Deleted) [[unlikely]] {
                        if (tombstone == m_bucket_count) [[likely]] tombstone = batch_start;
                    } else if (current_state == State::Empty) [[likely]] {
                        return tombstone != m_bucket_count ? tombstone : batch_start;
                    }
                }
                
                i = (i + 8) & mask;
                if (i == start) [[unlikely]] break;
            }

            return tombstone;
        }
#else
        // Original implementations for non-ARM platforms
        template <typename K>
        size_type find_key(const K& key) const {
            if (m_bucket_count == 0) [[unlikely]] return m_bucket_count;

            const size_type mask = m_bucket_count - 1;
            const size_type k = static_cast<size_type>(m_hasher(key));
            size_type i = k & mask;
            const size_type start = i;

            const std::uint32_t* flags = m_flags;
            const pointer buckets = m_buckets;
            constexpr size_type flags_mask = FLAGS_PER_U32 - 1;

            size_type word_idx = i >> 4;
            std::uint32_t flag_word = flags[word_idx];

            while (true) {
                const size_type shift = (i & flags_mask) * 2;
                State current_state = static_cast<State>((flag_word >> shift) & 0b11);
                if (current_state == State::Occupied) [[likely]] {
                    if (m_key_equal(buckets[i].first, key)) [[likely]] {
                        return i;
                    }
                } else if (current_state == State::Empty) [[unlikely]] {
                    return m_bucket_count;
                }

                i = (i + 1) & mask;
                if (i == start) [[unlikely]] break;

                const size_type new_word_idx = i >> 4;
                if (new_word_idx != word_idx) [[unlikely]] {
                    word_idx = new_word_idx;
                    flag_word = flags[word_idx];
                }
            }
            return m_bucket_count;
        }

        template <typename K>
        size_type find_insert_slot(const K& key) const {
            const size_type mask = m_bucket_count - 1;
            const size_type k = static_cast<size_type>(m_hasher(key));
            size_type i = k & mask;
            const size_type start = i;
            size_type tombstone = m_bucket_count;

            const std::uint32_t* flags = m_flags;
            constexpr size_type flags_mask = FLAGS_PER_U32 - 1;

            size_type word_idx = i >> 4;
            std::uint32_t flag_word = flags[word_idx];

            while (true) {
                const size_type shift = (i & flags_mask) * 2;
                State current_state = static_cast<State>((flag_word >> shift) & 0b11);
                if (current_state == State::Deleted) [[unlikely]] {
                    if (tombstone == m_bucket_count) [[likely]] tombstone = i;
                } else if (current_state == State::Empty) [[likely]] {
                    return tombstone != m_bucket_count ? tombstone : i;
                }

                i = (i + 1) & mask;
                if (i == start) [[unlikely]] break;

                const size_type new_word_idx = i >> 4;
                if (new_word_idx != word_idx) [[unlikely]] {
                    word_idx = new_word_idx;
                    flag_word = flags[word_idx];
                }
            }

            return tombstone;
        }
#endif

#ifdef __ARM_NEON
        // NEON-accelerized flag initialization
        void initialize_flags_neon(std::uint32_t* flags, size_type flag_array_size) {
            const uint32x4_t empty_pattern = vdupq_n_u32(EMPTY_FLAGS_PATTERN);
            size_type i = 0;
            
            // Process 4 flag words at a time using NEON
            for (; i + 4 <= flag_array_size; i += 4) {
                vst1q_u32(flags + i, empty_pattern);
            }
            
            // Handle remaining flags
            for (; i < flag_array_size; ++i) {
                flags[i] = EMPTY_FLAGS_PATTERN;
            }
        }
#endif

        void rehash_internal(size_type new_n_buckets) {
            if (new_n_buckets == 0) [[unlikely]] {
                clear();
                deallocate_storage();
                return;
            }

            new_n_buckets = roundup32(new_n_buckets);
            if (new_n_buckets <= m_bucket_count) [[unlikely]] return;

            pointer old_buckets = m_buckets;
            std::uint32_t* old_flags = m_flags;
            size_type old_n_buckets = m_bucket_count;

            m_buckets = std::allocator_traits<allocator_type>::allocate(m_allocator, new_n_buckets);

            FlagAlloc flag_alloc;
            const size_type flag_array_size = (new_n_buckets + FLAGS_PER_U32 - 1) / FLAGS_PER_U32;
            m_flags = flag_alloc.allocate(flag_array_size);

#ifdef __ARM_NEON
            // Use NEON for faster flag initialization
            initialize_flags_neon(m_flags, flag_array_size);
#else
            std::memset(m_flags, 0xaa, flag_array_size * sizeof(std::uint32_t));
#endif

            m_bucket_count = new_n_buckets;
            m_size = 0;
            m_occupied = 0;

            if (old_buckets) [[likely]] {
                auto old_get_state = [&](size_type idx) -> State {
                    const size_type word_index = idx / FLAGS_PER_U32;
                    const size_type shift = (idx % FLAGS_PER_U32) * 2;
                    std::uint32_t w = old_flags[word_index];
                    return static_cast<State>((w >> shift) & 0b11);
                };

#ifdef __ARM_NEON
                // Process in batches for better cache locality
                constexpr size_type BATCH_SIZE = 8;
                for (size_type batch_start = 0; batch_start < old_n_buckets; batch_start += BATCH_SIZE) {
                    size_type batch_end = std::min(batch_start + BATCH_SIZE, old_n_buckets);
                    
                    // Prefetch next batch
                    if (batch_start + BATCH_SIZE < old_n_buckets) {
                        __builtin_prefetch(old_buckets + batch_start + BATCH_SIZE, 0, 1);
                    }
                    
                    for (size_type i = batch_start; i < batch_end; ++i) {
                        if (old_get_state(i) == State::Occupied) [[likely]] {
                            size_type slot = find_insert_slot(old_buckets[i].first);
                            std::allocator_traits<allocator_type>::construct(m_allocator, m_buckets + slot,
                                                                             std::move(old_buckets[i]));
                            set_state(slot, State::Occupied);
                            m_size++;
                            m_occupied++;
                            std::allocator_traits<allocator_type>::destroy(m_allocator, old_buckets + i);
                        }
                    }
                }
#else
                for (size_type i = 0; i < old_n_buckets; ++i) {
                    if (old_get_state(i) == State::Occupied) [[likely]] {
                        size_type slot = find_insert_slot(old_buckets[i].first);
                        std::allocator_traits<allocator_type>::construct(m_allocator, m_buckets + slot,
                                                                         std::move(old_buckets[i]));
                        set_state(slot, State::Occupied);
                        m_size++;
                        m_occupied++;
                        std::allocator_traits<allocator_type>::destroy(m_allocator, old_buckets + i);
                    }
                }
#endif
                
                std::allocator_traits<allocator_type>::deallocate(m_allocator, old_buckets, old_n_buckets);
                const size_type old_flag_array_size = (old_n_buckets + FLAGS_PER_U32 - 1) / FLAGS_PER_U32;
                flag_alloc.deallocate(old_flags, old_flag_array_size);
            }
        }

        void destroy_elements() noexcept {
            if (!m_buckets) [[unlikely]] return;
            
#ifdef __ARM_NEON
            // Process in larger batches for better performance
            constexpr size_type BATCH_SIZE = 16;
            for (size_type i = 0; i < m_bucket_count; i += BATCH_SIZE) {
                size_type batch_end = std::min(i + BATCH_SIZE, m_bucket_count);
                
                // Check batch states efficiently
                for (size_type j = i; j < batch_end; ++j) {
                    if (is_occupied(j)) [[unlikely]] {
                        std::allocator_traits<allocator_type>::destroy(m_allocator, m_buckets + j);
                    }
                }
            }
#else
            for (size_type i = 0; i < m_bucket_count; ++i) {
                if (is_occupied(i)) [[unlikely]] {
                    std::allocator_traits<allocator_type>::destroy(m_allocator, m_buckets + i);
                }
            }
#endif
        }

        void deallocate_storage() noexcept {
            if (m_buckets) [[likely]] {
                std::allocator_traits<allocator_type>::deallocate(m_allocator, m_buckets, m_bucket_count);
                m_buckets = nullptr;
            }
            if (m_flags) [[likely]] {
                FlagAlloc flag_alloc;
                const size_type flag_array_size = (m_bucket_count + FLAGS_PER_U32 - 1) / FLAGS_PER_U32;
                flag_alloc.deallocate(m_flags, flag_array_size);
                m_flags = nullptr;
            }
            m_bucket_count = 0;
            m_size = 0;
            m_occupied = 0;
        }

    public:
        unordered_map() noexcept(noexcept(hasher()) && noexcept(key_equal()) && noexcept(allocator_type())) {}

        template <class InputIt>
        unordered_map(InputIt first, InputIt last) {
            insert(first, last);
        }

        unordered_map(const unordered_map& other)
            : m_max_load_factor(other.m_max_load_factor), m_hasher(other.m_hasher), m_key_equal(other.m_key_equal),
              m_allocator(
                  std::allocator_traits<allocator_type>::select_on_container_copy_construction(other.m_allocator)) {
            if (other.m_size > 0) [[likely]] {
                rehash_internal(static_cast<size_type>(other.m_size / m_max_load_factor) + 1);
                for (const auto& val : other)
                    insert(val);
            }
        }

        unordered_map(unordered_map&& other) noexcept
            : m_buckets(std::exchange(other.m_buckets, nullptr)), m_flags(std::exchange(other.m_flags, nullptr)),
              m_bucket_count(std::exchange(other.m_bucket_count, 0)), m_size(std::exchange(other.m_size, 0)),
              m_occupied(std::exchange(other.m_occupied, 0)), m_max_load_factor(other.m_max_load_factor),
              m_hasher(std::move(other.m_hasher)), m_key_equal(std::move(other.m_key_equal)),
              m_allocator(std::move(other.m_allocator)) {}

        ~unordered_map() {
            destroy_elements();
            deallocate_storage();
        }

        unordered_map& operator=(const unordered_map& other) {
            if (this == &other) [[unlikely]] return *this;
            clear();
            m_hasher = other.m_hasher;
            m_key_equal = other.m_key_equal;
            if (std::allocator_traits<allocator_type>::propagate_on_container_copy_assignment::value) [[unlikely]] {
                m_allocator = other.m_allocator;
            }
            reserve(other.m_size);
            insert(other.begin(), other.end());
            return *this;
        }

        unordered_map& operator=(unordered_map&& other) noexcept {
            if (this == &other) [[unlikely]] return *this;
            destroy_elements();
            deallocate_storage();
            m_buckets = std::exchange(other.m_buckets, nullptr);
            m_flags = std::exchange(other.m_flags, nullptr);
            m_bucket_count = std::exchange(other.m_bucket_count, 0);
            m_size = std::exchange(other.m_size, 0);
            m_occupied = std::exchange(other.m_occupied, 0);
            m_hasher = std::move(other.m_hasher);
            m_key_equal = std::move(other.m_key_equal);
            m_allocator = std::move(other.m_allocator);
            return *this;
        }

        iterator begin() noexcept { return iterator(this, 0); }
        const_iterator begin() const noexcept { return const_iterator(this, 0); }
        const_iterator cbegin() const noexcept { return const_iterator(this, 0); }
        iterator end() noexcept { return iterator(this, m_bucket_count); }
        const_iterator end() const noexcept { return const_iterator(this, m_bucket_count); }
        const_iterator cend() const noexcept { return const_iterator(this, m_bucket_count); }

        bool empty() const noexcept { return m_size == 0; }
        size_type size() const noexcept { return m_size; }
        size_type max_size() const noexcept { return std::allocator_traits<allocator_type>::max_size(m_allocator); }

        void clear() noexcept {
            destroy_elements();
            if (m_flags) [[likely]] {
                const size_type flag_array_size = (m_bucket_count + FLAGS_PER_U32 - 1) / FLAGS_PER_U32;
#ifdef __ARM_NEON
                initialize_flags_neon(m_flags, flag_array_size);
#else
                std::memset(m_flags, 0xaa, flag_array_size * sizeof(std::uint32_t));
#endif
            }
            m_size = 0;
            m_occupied = 0;
        }

        template <class... Args>
        std::pair<iterator, bool> emplace(Args&&... args) {
            if (m_bucket_count == 0 || m_occupied + 1 > m_bucket_count * m_max_load_factor) [[unlikely]] {
                rehash_internal(m_bucket_count > 0 ? m_bucket_count * 2 : 4);
            }
            value_type temp_val(std::forward<Args>(args)...);
            size_type index = find_key(temp_val.first);

            if (index != m_bucket_count) [[unlikely]] {
                return {iterator(this, index), false};
            }

            size_type slot = find_insert_slot(temp_val.first);
            bool was_empty = get_state(slot) == State::Empty;

            std::allocator_traits<allocator_type>::construct(m_allocator, m_buckets + slot, std::move(temp_val));
            set_state(slot, State::Occupied);
            m_size++;
            if (was_empty) [[likely]] m_occupied++;

            return {iterator(this, slot), true};
        }

        std::pair<iterator, bool> insert(const value_type& value) { return emplace(value); }
        std::pair<iterator, bool> insert(value_type&& value) { return emplace(std::move(value)); }
        template <class InputIt>
        void insert(InputIt first, InputIt last) {
            for (; first != last; ++first)
                emplace(*first);
        }

        iterator erase(const_iterator pos) {
            size_type index = pos.m_index;
            if (index >= m_bucket_count || !is_occupied(index)) [[unlikely]] return end();

            std::allocator_traits<allocator_type>::destroy(m_allocator, m_buckets + index);
            set_state(index, State::Deleted);
            --m_size;

            return ++iterator(this, index);
        }

        size_type erase(const key_type& key) {
            size_type index = find_key(key);
            if (index != m_bucket_count) [[likely]] {
                erase(const_iterator(this, index));
                return 1;
            }
            return 0;
        }

        void swap(unordered_map& other) noexcept {
            using std::swap;
            swap(m_buckets, other.m_buckets);
            swap(m_flags, other.m_flags);
            swap(m_bucket_count, other.m_bucket_count);
            swap(m_size, other.m_size);
            swap(m_occupied, other.m_occupied);
            swap(m_max_load_factor, other.m_max_load_factor);
            swap(m_hasher, other.m_hasher);
            swap(m_key_equal, other.m_key_equal);
            if (std::allocator_traits<allocator_type>::propagate_on_container_swap::value) [[unlikely]] {
                swap(m_allocator, other.m_allocator);
            }
        }

        mapped_type& at(const key_type& key) {
            size_type index = find_key(key);
            if (index == m_bucket_count) [[unlikely]] throw std::out_of_range("unordered_map::at");
            return m_buckets[index].second;
        }
        const mapped_type& at(const key_type& key) const {
            size_type index = find_key(key);
            if (index == m_bucket_count) [[unlikely]] throw std::out_of_range("unordered_map::at");
            return m_buckets[index].second;
        }

        mapped_type& operator[](const key_type& key) {
            if (m_bucket_count == 0 || m_occupied + 1 > m_bucket_count * m_max_load_factor) [[unlikely]] {
                rehash_internal(m_bucket_count > 0 ? m_bucket_count * 2 : 4);
            }
            size_type index = find_key(key);
            if (index != m_bucket_count) [[unlikely]] return m_buckets[index].second;

            size_type slot = find_insert_slot(key);
            bool was_empty = get_state(slot) == State::Empty;

            std::allocator_traits<allocator_type>::construct(m_allocator, m_buckets + slot, key, mapped_type{});
            set_state(slot, State::Occupied);
            m_size++;
            if (was_empty) [[likely]] m_occupied++;

            return m_buckets[slot].second;
        }

        size_type count(const key_type& key) const { return find_key(key) != m_bucket_count ? 1 : 0; }
        iterator find(const key_type& key) {
            size_type index = find_key(key);
            return index == m_bucket_count ? end() : iterator(this, index);
        }
        const_iterator find(const key_type& key) const {
            size_type index = find_key(key);
            return index == m_bucket_count ? cend() : const_iterator(this, index);
        }
        std::pair<iterator, iterator> equal_range(const key_type& key) {
            iterator it = find(key);
            return {it, (it == end() ? it : std::next(it))};
        }
        std::pair<const_iterator, const_iterator> equal_range(const key_type& key) const {
            const_iterator it = find(key);
            return {it, (it == cend() ? it : std::next(it))};
        }

        size_type bucket_count() const noexcept { return m_bucket_count; }
        size_type bucket(const key_type& key) const {
            return m_bucket_count == 0 ? 0 : m_hasher(key) & (m_bucket_count - 1);
        }
        size_type bucket_size(size_type n) const { return (n < m_bucket_count && is_occupied(n)) ? 1 : 0; }

        float load_factor() const noexcept {
            return m_bucket_count == 0 ? 0.0f : static_cast<float>(m_size) / m_bucket_count;
        }
        float max_load_factor() const noexcept { return m_max_load_factor; }
        void max_load_factor(float ml) { m_max_load_factor = ml; }
        void rehash(size_type count) {
            if (count > m_bucket_count) [[likely]] rehash_internal(count);
        }
        void reserve(size_type count) {
            if (count > 0) [[likely]] {
                rehash_internal(static_cast<size_type>(count / m_max_load_factor) + 1);
            }
        }
    };

} // namespace FastSTL
