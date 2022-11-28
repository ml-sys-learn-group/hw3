//
// Created by hzx on 22-11-9.
//

#ifndef NDARRAY_BACKEND_CPU_H
#define NDARRAY_BACKEND_CPU_H

namespace needle {
    namespace cpu {

#define ALIGNMENT 256
#define TILE 8
        typedef float scalar_t;
        const size_t ELEM_SIZE = sizeof(scalar_t);

        /**
         * This is a utility structure for maintaining an array aligned to ALIGNMENT boundaries in
         * memory.  This alignment should be at least TILE * ELEM_SIZE, though we make it even larger
         * here by default.
         */
        struct AlignedArray {
            AlignedArray(const size_t size) {
                int ret = posix_memalign((void **) &ptr, ALIGNMENT, size * ELEM_SIZE);
                if (ret != 0) throw std::bad_alloc();
                this->size = size;
            }

            ~AlignedArray() { free(ptr); }

            size_t ptr_as_int() { return (size_t) ptr; }

            scalar_t *ptr;
            size_t size;
        };

        /**
         * fill the values of an aligned array with val
         * @param out
         * @param val
         */
        void Fill(AlignedArray *out, scalar_t val);

        /**
         * compact array
         * @param a
         * @param out
         * @param shape
         * @param strides
         * @param offset
         */
        void Compact(const AlignedArray &a, AlignedArray *out, std::vector<uint32_t> shape,
                     std::vector<uint32_t> strides, size_t offset);


        /**
         * get slice according to index
         * @param index
         * @param compat_strides
         * @param slice
         */
        void GetGridIndex(size_t index,
                          const std::vector<uint32_t> &compat_strides,
                          std::vector<uint32_t> &grid_index);


        /**
         * get the storage index
         * @param grid_index
         * @param strides
         * @return
         */
        size_t GetStorageIndex(const std::vector<uint32_t>& grid_index,
                               const std::vector<uint32_t>& strides);


        /**
         *
         * @param shape
         * @param compat_strides
         */
        void GetCompactStrides(const std::vector<uint32_t>& shape,
                               const size_t& elem_size,
                               std::vector<uint32_t>& compat_strides);


        /**
         * get the whole element size
         * @param shape
         * @return
         */
        size_t GetElemSize(const std::vector<uint32_t>& shape);


        /**
         * reduce max
         * @param a
         * @param out
         * @param reduce_size
        */
        void ReduceMax(const AlignedArray& a, AlignedArray* out, size_t reduce_size);

        /**
         * reduce sum
         * @param a
         * @param out
         * @param reduce_size
        */
        void ReduceSum(const AlignedArray& a, AlignedArray* out, size_t reduce_size);
    }
}
#endif // NDARRAY_BACKEND_CPU_H
