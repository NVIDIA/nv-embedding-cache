/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
    \file embed_cache.h
    embedding cache library interface
*/
 
#pragma once

// need to include this for cudaCalls can we move this to another place
#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <cstdio>
#include <cstdarg>
#include <unordered_map>
#include <algorithm>

#include <logging.hpp>
#include <allocator.hpp>

#define EC_MAX_STRING_BUF 1024

#define ECERROR_SUCCESS 0x0 /**< all operation finished successfully */ 
#define ECERROR_INVALID_ARGUMENT 0x1 /**< one of the functions arguments was illegal for instance a ptr set to null */ 
#define ECERROR_MEMORY_ALLOCATION 0x2 /**< returned when a memory allocation failed */ 
#define ECERROR_CUDA_ERROR 0x3 /**< returned when a cache recived a cuda driver error */ 
#define ECERROR_FREE_ERROR 0x4
#define ECERROR_NOT_IMPLEMENTED 0x5  /**< returned when a function wasn't implemented */ 
#define ECERROR_STALE_MODIFY_CONTEXT 0x6 /**< returned when a ModifyContext was calculated on an old state and no modify operation can be done on it */
#define ECERROR_BAD_ALLOCATOR 0x7 /**< return during init function if allocator is nullptr */
#define ECERROR_BAD_LOGGER 0x8 /**< return during init function if logger is nullptr */
#define ECERROR_MEMORY_ALLOCATED_TO_CACHE_TOO_SMALL 0x9 /**< return in init if memory allowed for cache is too small to initialize internal buffers */
#define ECERROR_BAD_KEY_TAG_CONFIG 0xA /**< return when there is either a mismatch between Tag and Key or an unsupported pair */

#ifndef EC_THROW
#define EC_THROW(err) \
do { \
    throw ECException(err, "%s detected at: %s:%d", #err, __FILE__, __LINE__); \
} while (false)
#endif

#ifndef CACHE_CUDA_ERR_CHK_AND_THROW
#define CACHE_CUDA_ERR_CHK_AND_THROW(ans) \
do { \
    const auto res = ans; \
    if (res != cudaSuccess) { \
        throw ECException(ECERROR_CUDA_ERROR, "%s Error detected at: %s:%d", cudaGetErrorName(res), __FILE__, __LINE__); \
    } \
} while (false)
#endif

#ifndef CHECK_ERR_AND_THROW
#define CHECK_ERR_AND_THROW(ans) \
do { \
    const auto res = ans; \
    if ((res) != ECERROR_SUCCESS) { \
        throw ECException(res, "ECError %d detected at: %s:%d", res , __FILE__, __LINE__); \
    } \
} while (false)
#endif

#ifndef CACHE_ALIGN
#define CACHE_ALIGN_MASK(x,mask)    (((x)+(mask))&~(mask))
#define CACHE_ALIGN(x,a)              CACHE_ALIGN_MASK(x,(a)-1)
#endif

/**
    \namespace nve
    all embedding cache objects live in this namespace
*/
namespace nve {


/** \brief Enumrating all the implemenation type, used to get the runtime type
*/ 
enum CACHE_IMPLEMENTATION_TYPE
{
    API,  
    SET_ASSOCIATIVE_HOST_METADATA,
    SET_ASSOCIATIVE_DEVICE_ONLY,

    NUM_IMPLEMENTATION_TYPES,
};

enum DataTypeFormat
{
    DATATYPE_FP16,
    DATATYPE_FP32,
    DATATYPE_INT8_SCALED,

    NUM_DATA_TYPES_FORMATS,
};

/**
    \typedef ECError
    The type of the returned error code
*/
typedef uint32_t ECError;

/**
    class representing all exception thrown by internal functions
*/
class ECException: public std::exception
{
public:
    /**
        A constructor

        @param  the error code 
    */

    ECException(ECError err, const char* format, ...) : m_err(err)
    {
        std::va_list args;
        va_start(args, format);
        std::vsnprintf(buf, EC_MAX_STRING_BUF, format, args);
        va_end(args);
    }
    
    ECException(ECError err) : m_err(err) 
    {
        std::snprintf(buf, EC_MAX_STRING_BUF, "ECError %d occured\n", m_err);
    }

    /**
        return a stringified descrpition of the error occured
    */
    virtual const char* what() const throw()
    {
        return buf;
    }

    ECError m_err;
private:
    char buf[EC_MAX_STRING_BUF] = {0};

};

/** \brief synchronization object interface
 * 
 *  Interface which allow the user implement a custom synchronization the application required while maintaining the following idiom. 
    When calling event_record we logically place a marker at time t_0 – Can be thought of as the time event_record() returned. When calling event_wait_stream() all further execution on stream s needs to wait until all Cache Lookups happened before time t_0 finished.
    For simplicy an default implementation is supplied see DefaultECEvent
*/ 
class IECEvent
{
public:
    /** \brief Mark a time point for synchronization 
    * 
    *  When this function is called we logically place a time point on all the inferences timeline. Practically we can think of an event placed on all the streams that are performing inference with a specific cache 
    * 
    * @return #ECERROR_SUCCESS on success
    */ 
    virtual ECError event_record() = 0;

    /** \brief wait on previously placed marker
    * 
    *  When this function is called stream s is stalled untill all inferences that started before the previously placed marker are done. practically this can be thought of as cudaStreamWaitEvent()
    * 
    * @return #ECERROR_SUCCESS on success
    */ 
    virtual ECError event_wait_stream(cudaStream_t stream) = 0;

    /** \brief D'tor
    * 
    *  Default destructor
    * 
    */ 
   virtual ~IECEvent() = default;
};

template<typename IndexT>
class DefaultHistogram
{
public:
    struct Bin
    {
        IndexT key;
        float count;
        const int8_t* data;
    };

    DefaultHistogram(const IndexT* indices, size_t sz, const int8_t* data, size_t stride_in_bytes, bool is_linear)
    {
        bins_.reserve(sz);
        key_bin_map_.reserve(sz);
        for (size_t i = 0; i < sz; i++)
        {
            auto idx = indices[i];
            if (key_bin_map_.count(idx) == 0)
            {
                uint64_t offset = (is_linear ? idx : i);
                Bin new_bin = {idx, 1, data + offset * stride_in_bytes};
                key_bin_map_[idx] = static_cast<IndexT>(bins_.size());
                bins_.push_back(new_bin);
            }
            else
            {
                auto curr_bin_loc = key_bin_map_[idx];
                Bin& bin = bins_[curr_bin_loc];
                bin.count += 1;
            }
        }
        std::sort(bins_.begin(), bins_.end(), [](const Bin& a, const Bin& b) { return a.count > b.count;});

        for (auto bin : bins_)
        {
            keys_.push_back(bin.key);
            // Normalize priority by total number of elements
            priority_.push_back(float(bin.count) / float(sz));
            data_.push_back(bin.data);
        }
    }
    size_t get_num_bins() const
    {
        return bins_.size();
    }

    float* get_priority() 
    {
        return priority_.data();
    }

    IndexT* get_keys() 
    {
        return keys_.data();
    }

    const int8_t* const* get_data()
    {
        return data_.data();
    }

private:
    std::vector<Bin> bins_;
    std::vector<IndexT> keys_;
    std::vector<float> priority_;
    std::vector<const int8_t*> data_;
    std::unordered_map<IndexT, IndexT> key_bin_map_;
};

class DefaultECEvent : public IECEvent
{
public:
    template <typename T>
    DefaultECEvent(const T& streams)
    {
        for (auto& s : streams)
        {
            streams_.push_back(s);
            cudaEvent_t nEvent;
            CACHE_CUDA_ERR_CHK_AND_THROW(cudaEventCreateWithFlags(&nEvent, cudaEventDisableTiming));
            events_.push_back(nEvent);
        }
    }

    ~DefaultECEvent() override
    {
        for (cudaEvent_t event : events_)
        {
            if (cudaEventDestroy(event) != cudaSuccess) {
                // Not throwing an error in the destructor
                // Todo: report this failure some other way
            }
        }
    }

    ECError event_record() override
    {

        for (size_t i = 0; i < streams_.size(); i++)
        {
            CACHE_CUDA_ERR_CHK_AND_THROW(cudaEventRecord(events_[i], streams_[i]));
        } 
        return ECERROR_SUCCESS;
    }

    ECError event_wait_stream(cudaStream_t stream) override
    {

        for (size_t i = 0; i < events_.size(); i++)
        {
            CACHE_CUDA_ERR_CHK_AND_THROW(cudaStreamWaitEvent(stream, events_[i]));
        } 
        return ECERROR_SUCCESS;
    }

    const std::vector<cudaStream_t>& get_streams() const { return streams_; }
    const std::vector<cudaEvent_t>& get_events() const { return events_; }

private:
    std::vector<cudaStream_t> streams_;
    std::vector<cudaEvent_t> events_;
};

/** \brief Enumrating all the performance metric type
*/ 
enum PerformanceMerticTypes
{
    MERTIC_COUNT_MISSES, 
    
    NUM_PERFORMANCE_METRIC_TYPES,
};

struct CacheAllocationSize
{
    size_t device_allocation_size{0};
    size_t host_allocation_size{0};
};

/** \brief Handle to modify context
*/ 
struct ModifyContextHandle
{
    uint64_t handle{0};
};

/** \brief Handle to lookup context
*/ 
struct LookupContextHandle
{
    uint64_t handle{0};
};

/** \brief Performance metric
*/ 
struct PerformanceMetric
{
    int64_t* d_val {nullptr};
    PerformanceMerticTypes type {NUM_PERFORMANCE_METRIC_TYPES};
};

/**
 * \brief Interface for embedding cache
 * 
 * All cache implementation implement this interface to allow for unifed API. this class performs the management of the Embedding Cache on GPU side and performs the major operations of Lookup and Modify
*/
template<typename IndexT>
class EmbedCacheBase
{
public:
    /**
     * \brief C'tor
     * 
     * Constructs the baseclass all derived class need to supply the allocator and a logger and the type. The allocator is used to allocate memory, Logger is for logging purposes (not wood). And type might be used for various action to cast down to explicit implementation.
     * No memory should be allocated during construction only on Init();
     * 
     * @param allocator – Pointer to an implementation of the Allocator interface
       @param logger – Pointer to an implementation of the Logger interface
       @param type - The type of the implementation see nve::CACHE_IMPLEMENTATION_TYPE
    */
    EmbedCacheBase(Allocator* allocator, Logger* logger, CACHE_IMPLEMENTATION_TYPE type) : allocator_(allocator), logger_(logger), type_(type){}


    /**
     * \brief D'tor
    */
    virtual ~EmbedCacheBase() = default;

    /**
     * \brief Placeholder struct for interfacing with a user kernel
     * 
     * All implementation of this interface have a struct CacheData allow for templating of user kernels
    */
    struct CacheData
    {

    };

    /**
     * \brief Returns the type of implementation
    */
    CACHE_IMPLEMENTATION_TYPE get_type() const { return type_; }

    /** \brief Initialize the cache, allocate all the internal buffers, based on specific requirements by the implementation, No actions can be performed on the cache until it is initialized.
     * 
     * @return #ECERROR_SUCCESS on success, might return different values based on implementation
     * May return #ECERROR_MEMORY_ALLOCATED_TO_CACHE_TOO_SMALL if size allowed for cache doesn't meet minimum requirement
     * May return #ECERROR_MEMORY_ALLOCATION in allocation errors
    */
    virtual ECError init() = 0;

    
    // cache Accessors
    /**
     * \brief Create a LookupContext and return an handle to in out_handle 
     * 
     * The lookup context is used for all lookup operation and provides a context to the cache, two concurrent lookups in the cache should use different contexts. 
     * Performance metrics supplied here bind the context to perform metrics accumulation on the metrics supplied in the metrics array of size num_metrics. 
     * Allocates the memory required via the allocator 
     * 
     * @param out_handle – The handle to the lookupContext is returned here
     * @param metrics – An array of PerformanceMetrics to be bound to this context all lookup operations done with this context will accumulate the metrics specified here
     * @param num_metrics – Size of the metrics array
     * 
     * @return #ECERROR_SUCCESS on success, might return different values based on implementation
     * May return #ECERROR_MEMORY_ALLOCATION in allocation errors
    */
    virtual ECError lookup_context_create(LookupContextHandle& out_handle, const PerformanceMetric* metrics, size_t num_metrics) const = 0;

    /**
     * \brief Destroys and free the memory of the LookupContext handled by handle. 
     * 
     * @param handle – context to be destroyed
     * 
     * @return #ECERROR_SUCCESS on success, might return different values based on implementation
    */
    virtual ECError lookup_context_destroy(LookupContextHandle& handle) const = 0;

    /**
     * \brief cast the LookupContextHandle to its actual type
     * 
     * For Templatization of functions
    */
    CacheData get_cache_data(const LookupContextHandle* hangle_ptr) const;

    // return dense representation of the indices
    /**
     * 
     * \brief Performs a lookup without resolving misses
     * 
     * In the context handled by lookup_handle Performs a lookup from a cache of len keys placed in device memory pointed by d_keys, returns the values which resides in the cache internal buffers, In the device memory pointed by d_values. 
     * If d_keys[i] isn’t stored in the cache, then append i to the device memory pointed by d_missing_index, and append d_keys[i] to the device memory pointed by d_missing_keys, the number of missing keys will be returned in d_missing_len.
     * All the performed key queries are related to table index curr_table.
     * Pseudocode for Lookup
     * 1	:	d_missing_len = 0
     * 2	:	For i = 1 .. len
     * 3	:	   If d_keys[i] in cache 
     * 4	:	       (d_values + i*stride) = cache[d_keys[i]]
     * 5	:	   else
     * 6	:	        d_missing_keys[d_missing_len] = d_keys[i]
     * 7	:	        d_missing_index[d_missing_len] = i
     * 8	:	        d_missing_len++
     * 
     * @param lookup_handle – The handle to the lookupContext to perform the lookup here
     * @param d_keys – Pointer to device side memory that holds the keys to query
     * @param len – Size of the d_keys array
     * @param d_values – Pointer to device side memory to write the matching values of the keys that reside in the cache
     * @param d_missing_index – Pointer to device side memory to write the index of the missing key
     * @param d_missing_key - Pointer to device side memory to write the missing key
     * @param d_missing_len – Pointer to device side memory to write the length of the missing key array
     * @param curr_table – Index of the table to perform the query on 
     * @param stride – The length in bytes between two consecutive “rows” in d_values array,
     * @param stream -  A cuda stream to perform the operation on

     * @return #ECERROR_SUCCESS on success, might return different values based on implementation
    */
    virtual ECError lookup(const LookupContextHandle& lookup_handle, const IndexT* d_keys, const size_t len,
                                            int8_t* d_values, uint64_t* d_missing_index,
                                            IndexT* d_missing_keys, size_t* d_missing_len,
                                            uint32_t curr_table, size_t stride, cudaStream_t stream) = 0;


    /**
     * 
     * \brief Performs a lookup without resolving misses return hit mask for future processing
     * 
     * In the context handled by lookup_handle Performs a lookup from a cache of len keys placed in device memory pointed by d_keys, returns the values which resides in the cache internal buffers, In the device memory pointed by d_values. 
     * If d_keys[i] isn’t stored in the cache, then hit_mask i'th bit will be set to 0.
     * All the performed key queries are related to table index curr_table.
     * 
     * @param lookup_handle – The handle to the lookupContext to perform the lookup here
     * @param d_keys – Pointer to device side memory that holds the keys to query
     * @param len – Size of the d_keys array
     * @param d_values – Pointer to device side memory to write the matching values of the keys that reside in the cache
     * @param d_hit_mask – Pointer to device side memory to write hits, i bit == 1 iff d_keys[i] in cache
     * @param curr_table – Index of the table to perform the query on 
     * @param stride – The length in bytes between two consecutive “rows” in d_values array,
     * @param stream -  A cuda stream to perform the operation on

     * @return #ECERROR_SUCCESS on success, might return different values based on implementation
    */
    virtual ECError lookup(const LookupContextHandle& lookup_handle, const IndexT* d_keys, const size_t len,
                                            int8_t* d_values, uint64_t* d_hit_mask,
                                            uint32_t curr_table, size_t stride, cudaStream_t stream) = 0;

    /**
     * \brief Performs a lookup
     *  
     * Performs a lookup from the cache, of len keys placed in device memory pointed by d_keys, if d_keys[i] return value placed in cache internal buffers to device memory pointed by d_values, else return the value in d_table + rowSize*d_keys[i] in d_values. 
     * This perform a similar operation to a gather operation for a table backed up cache, and differs from the previous lookup operation, by the fact that this operation can resolve cache misses on its own. 
     * Pseudocode for Lookup
     * 1	:	For i = 1 .. len
     * 2	:	   If d_keys[i] in cache 
     * 3	:	       (d_values + i*stride) = cache[d_keys[i]]
     * 4	:	   else
     * 5	:	        (d_values + i*stride) = d_table[d_keys[i]]
     * 
     * @param lookup_handle – The handle to the lookupContext to perform the lookup here.
     * @param d_keys – Pointer to device side memory that holds the keys to query.
     * @param len – Size of the d_keys array.
     * @param d_values – Pointer to device side memory to write the matching values of the keys that reside in the cache.
     * @param d_table – Pointer to device accessible memory where the embedding table reside, all missed keys will be resolved by this base pointer. This assumes all keys are valid values in the d_table.
     * @param curr_table – Index of the table to perform the query on. 
     * @param stride – The length in bytes between two consecutive “rows” in d_values array.
     * @param stream -  A cuda stream to perform the operation on.
     * 
     * @return #ECERROR_SUCCESS on success, might return different values based on implementation
    */

    virtual ECError lookup(const LookupContextHandle& lookup_handle, const IndexT* d_keys, const size_t len,
                                            int8_t* d_values, const int8_t* d_table, uint32_t curr_table, size_t stride, cudaStream_t stream) = 0;


    /**
     * \brief Performs a lookup
     *  
     * This API defers from previous lookup calls by performing multiple phases; First it will perform residency check, sort the results and then it will perform the gather.
     * This API is mainly benficial for very large tables.
     * 
     * User first need to query the size of the required auxiliry buffer by passing nullptr in d_auxiliary_buffer param, the result will be returned via auxiliary_buffer_bytes.
     * 
     * @param lookup_handle – The handle to the lookupContext to perform the lookup here.
     * @param d_keys – Pointer to device side memory that holds the keys to query.
     * @param len – Size of the d_keys array.
     * @param d_values – Pointer to device side memory to write the matching values of the keys that reside in the cache.
     * @param d_table – Pointer to device accessible memory where the embedding table reside, all missed keys will be resolved by this base pointer. This assumes all keys are valid values in the d_table.
     * @param d_auxiliary_buffer - Pointer to device accessible memory for auxiliry calculation. The user need to allocatate the required buffer, by first querying the required size by calling same function with nullptr at this argument
     * @param auxiliary_buffer_bytes - Size of the auxiliry buffer; if d_auxiliary_buffer isn't nullptr the function will validate this is large enough, if d_auxiliary_buffer is nullptr the function will return the required size
     * @param curr_table – Index of the table to perform the query on. 
     * @param stride – The length in bytes between two consecutive “rows” in d_values array.
     * @param block_value - The size of the block that kernel will use.
     * @param stream -  A cuda stream to perform the operation on.
     * 
     * @return #ECERROR_SUCCESS on success, might return different values based on implementation
    */

    virtual ECError lookup_sort_gather(const LookupContextHandle& lookup_handle, const IndexT* d_keys, const size_t len,
                                            int8_t* d_values, const int8_t* d_table, int8_t* d_auxiliary_buffer, size_t& auxiliary_buffer_bytes, uint32_t curr_table, size_t stride, int64_t block_size, cudaStream_t stream) = 0;


    /**
     * \brief Performs an non synchronized update accumulate
     *  
     * Generally update accumulate require synchronization with concurent Modify operation or lookup operation currently in flight. But in some cases where the application can gurantee that no other operation are currently in flight
     * We can use this function to get an optimized version of the update accumulate. 
     * 
     * @param lookup_handle – The handle to the lookupContext to perform Update accumulate
     * @param modify_handle – Handle to modify context.
     * @param d_keys - keys to perform an update accumulate on if they are present in the cache storage
     * @param len – Size of the d_keys array.
     * @param d_values – Pointer to device accessiable memory where the deltas are stored
     * @param curr_table – Index of the table to perform the query on. 
     * @param stride – The length in bytes between two consecutive “rows” in d_values array.
     * @param input_format - dataType of the values stored in d_values.
     * @param output_format - the precision in which to perform the accumulation and store in the cache storage.
     * @param stream -  A cuda stream to perform the operation on.
     * 
     * @return #ECERROR_SUCCESS on success, might return different values based on implementation
    */
    virtual ECError update_accumulate_no_sync(const LookupContextHandle& lookup_handle, ModifyContextHandle& modify_handle, const IndexT* d_keys, const size_t len, const int8_t* d_values, uint32_t curr_table, size_t stride, 
                                            DataTypeFormat input_format, DataTypeFormat output_format, cudaStream_t stream) = 0;


    /**
     * \brief EXPERIMENTAL Performs an non synchronized update accumulate to cache and uvm table
     *  
     * Generally update accumulate require synchronization with concurent Modify operation or lookup operation currently in flight. But in some cases where the application can gurantee that no other operation are currently in flight
     * We can use this function to get an optimized version of the update accumulate. 
     * 
     * @param lookup_handle – The handle to the lookupContext to perform Update accumulate
     * @param modify_handle – Handle to modify context.
     * @param d_keys - keys to perform an update accumulate on if they are present in the cache storage
     * @param len – Size of the d_keys array.
     * @param d_values – Pointer to device accessiable memory where the deltas are stored
     * @param curr_table – Index of the table to perform the query on. 
     * @param stride – The length in bytes between two consecutive “rows” in d_values array.
     * @param input_format - dataType of the values stored in d_values.
     * @param output_format - the precision in which to perform the accumulation and store in the cache storage.
     * @param uvm - ptr to device accessiable uvm table to udpate
     * @param stream -  A cuda stream to perform the operation on.
     * 
     * @return #ECERROR_SUCCESS on success, might return different values based on implementation
    */
    virtual ECError update_accumulate_no_sync_fused(const LookupContextHandle& lookup_handle, ModifyContextHandle& modify_handle, const IndexT* d_keys, const size_t len, const int8_t* d_values, uint32_t curr_table, size_t stride, 
                                            DataTypeFormat input_format, DataTypeFormat output_format, int8_t* uvm, cudaStream_t stream) = 0;

    // performance 
    /**
     * \brief Create and allocate a performance metric based on type
     * @param out_metric – The crated metric will be stored here.
     * @param type – Type to create.
     * 
     * @return #ECERROR_SUCCESS on success, might return different values based on implementation 
    */
    virtual ECError performance_metric_create(PerformanceMetric& out_metric, PerformanceMerticTypes type) const = 0;

    /**
     * \brief Destroy and deallocate the Perfromancemetric metric
     * 
     * @param metric - The metric to be deallocated and destroyed
     * 
     * @return #ECERROR_SUCCESS on success, might return different values based on implementation 
    */
    virtual ECError performance_metric_destroy(PerformanceMetric& metric) const = 0;

    /**
     * \brief  Query the value stored in the performance metric and returns it in the memory location pointed by out_value. 
     * 
     * this operation require GPU execution and thus require a stream to perform on. The semantics of the value is determined by the type of the performancemetric. 
     * 
     * @note  The User needs to wait on stream finished execution before checking the result. 
     * 
     * @param metric – The performance metric to Query for value.
     * @param out_value – A pointer to a memory where the value will be placed.
     * @param stream – A stream to perform the GPU execution to get the required metric – usually a copy.
     * 
     * @return #ECERROR_SUCCESS on success, might return different values based on implementation
    */
    virtual ECError performance_metric_get_value(const PerformanceMetric& metric, int64_t* out_value, cudaStream_t stream) const = 0;

    /**
     * \brief Reset the value of metric to its default value based on type. 
     * 
     * @note Between subsequent lookups metrics are accumulated in counter. GPU execution is done on stream and require a synchronization before operation Is done
     * 
     * @param metric – The performance metric to rest.
     * @param stream – A stream to perform the GPU execution to get the required metric.
     * 
     * @return #ECERROR_SUCCESS on success, might return different values based on implementation
    */
    virtual ECError performance_metric_reset(PerformanceMetric& metric, cudaStream_t stream) const = 0;


    /**
     * \brief Creates a ModifyContext used by subsequent calls to any ModifyOps. 
     * 
     * ModifyContext acts as a private storage area allowing multiple threads performing any modifyOp. This function will allocate both deviceMemory and hostMemory through the Allocator interface. The amount of memory and type is determined by implementation and controlled by max_update_size parameter.
     * The max_update_size specfis the max number of keys allowed to modify in each call to Modify() 
     * Each ModifyContext can handle modification of a single table inside the cache.
     * 
     * @param out_handle – The returned created ModifyContextHandle.
     * @param max_update_size – Max number of keys allowed to be modified in each call using this context.
     * 
     * @return #ECERROR_SUCCESS on success, might return different values based on implementation
    */
    virtual ECError modify_context_create(ModifyContextHandle& out_handle, uint32_t max_update_size) const = 0;

     /**
     * \brief Destroy and deallocate the modifyContext in handle
     * 
     * @param handle – The context handle to deallocate     
     * 
     * @return #ECERROR_SUCCESS on success, might return different values based on implementation
    */
    virtual ECError modify_context_destroy(ModifyContextHandle& handle) const = 0;

    /** @name Insert
    * @brief Considering a representative set of keys with respective priorities and data, the cache will calculate which of those keys should be placed in the cache to maximize hit rate.
    * This may result in some keys being evicted from the cache.
    * @note Keys must be unique and sorted by priority (from highest to lowest)
    * @param modify_context_handle – Handle to the modifyContext to use 
    * @param keys – Array of keys/indices to try and place in the cache
    * @param priority - Array of key priority (higher is better)
    * @param pp_data – Array of pointers to data vectors for the given keys
    * @param num_keys - Number of keys in "keys"
    * @param table_index – The index of the table to perform this operation on
    * @param sync_event – a synchronization event instanton to use as a synchronization between all Modify, Lookup operations
    * @param stream – a cudaStream to be processed on
    * 
    * @return EC_SUCCESS on success, otherwise error type.
    */
    virtual ECError insert(
        ModifyContextHandle& modify_context_handle,
        const IndexT* keys,
        const float* priority,
        const int8_t* const* pp_data,
        size_t num_keys,
        uint32_t table_index,
        IECEvent* sync_event,
        cudaStream_t stream) = 0;
    
    /** @name Update
    * @brief Given a set of keys with respective data, the cache will overwrite the data vectors of keys already resident in the cache with the new provided data.
    * This will not change key residency in the cache (no new key will be introduced and no key will be evicted). Keys not resident in the cache will be ignored.
    * @note Keys must be unique
    * @param modify_context_handle – Handle to the modifyContext to use 
    * @param keys – Array of keys/indices to try and place in the cache
    * @param d_values – Pointer to device accessiable memory where the data vectors are stored
    * @param stride – The length in bytes between two consecutive “rows” in d_values array
    * @param num_keys - Number of keys in "keys"
    * @param table_index – The index of the table to perform this operation on
    * @param sync_event – a synchronization event instanton to use as a synchronization between all Modify, Lookup operations
    * @param stream – a cudaStream to be processed on
    * 
    * @return EC_SUCCESS on success, otherwise error type.
    */
    virtual ECError update(
        ModifyContextHandle& modify_context_handle,
        const IndexT* keys,
        const int8_t* d_values,
        int64_t stride,
        size_t num_keys,
        uint32_t table_index,
        IECEvent* sync_event,
        cudaStream_t stream) = 0;
   
    /** @name update_accumulate
    * @brief Given a set of keys with respective data, the cache will accumulate the data vectors of keys already resident in the cache with the existing data vectors.
    * This will not change key residency in the cache (no new key will be introduced and no key will be evicted). Keys not resident in the cache will be ignored.
    * @note Keys must be unique
    * @param modify_context_handle – Handle to the modifyContext to use 
    * @param keys – Array of keys/indices to try and place in the cache
    * @param d_values – Pointer to device accessiable memory where the deltas are stored
    * @param stride – The length in bytes between two consecutive “rows” in d_values array
    * @param num_keys - Number of keys in "keys"
    * @param table_index – The index of the table to perform this operation on
    * @param update_format - The data format of the data vectors in "pp_data"
    * @param cache_format - The data format of the data vectors in the cache
    * @param sync_event – a synchronization event instanton to use as a synchronization between all Modify, Lookup operations
    * @param stream – a cudaStream to be processed on
    * 
    * @return EC_SUCCESS on success, otherwise error type.
    * 
    * @note Not all update_format/cache_format combinations are supported.
    */
    virtual ECError update_accumulate(
        ModifyContextHandle& modify_context_handle,
        const IndexT* keys,
        const int8_t* d_values,
        int64_t stride,
        size_t num_keys,
        uint32_t table_index,
        DataTypeFormat update_format,
        DataTypeFormat cache_format,
        IECEvent* sync_event,
        cudaStream_t stream) = 0;

    /** @name Invalidate
    * @brief Remove a given a set of keys from the cache.
    * @param modify_context_handle – Handle to the modifyContext to use 
    * @param keys – Array of keys/indices to try and place in the cache
    * @param num_keys - Number of keys in "keys"
    * @param table_index – The index of the table to perform this operation on
    * @param sync_event – a synchronization event instanton to use as a synchronization between all Modify, Lookup operations
    * @param stream – a cudaStream to be processed on
    * 
    * @return EC_SUCCESS on success, otherwise error type.
    */
    virtual ECError invalidate(
        ModifyContextHandle& modify_context_handle,
        const IndexT* keys,
        size_t num_keys,
        uint32_t table_index,
        IECEvent* sync_event,
        cudaStream_t stream) = 0;

    /**
     * \brief Clear the contents of the cache, all following lookups will return a miss
     *
     * Invalidates the content of the cache effectively results in all queries to be a miss, data itself isn't overwritten.
     * This is a non asynchronious function and user is expected to sync with any other operation currently in flight before
     * call this API.
     * 
     * @param stream - a cuda stream to perform an neccessary cuda operations
     * 
     * @return EC_SUCCESS on success, Might return other values based on implementation
    */
    virtual ECError clear_cache(cudaStream_t stream) = 0;

    /**
    * \brief Return the keys (duplication included) that are stored in the cache, this is useful in debugging and might cause device sync
    * 
    * Return the keys (duplication included) that stored in the cache, in the user allocated buffer out_keys, the number of keys is returned in num_out_keys.
    * The buffer should be allocated with size of sizeof(IndexT)*get_max_num_embedding_vectors_in_cache()
    * The data is true for the moment on call, if there are any modify calls in flight results are undefined and it is up to the user to sync
    * 
    * @param lookup_context_handle - handle to the lookup context to use
    * @param out_keys - output user allocated buffer to populate with the stored keys 
    * @param num_out_keys - output return the number of keys populated in out_keys array
    *
    * @return EC_SUCCESS on success, Might return other values based on implementation
    */
    virtual ECError get_keys_stored_in_cache(const LookupContextHandle& lookup_context_handle, IndexT* out_keys, size_t& num_out_keys) const = 0;

    /**
     * \brief Start a custom flow
     * 
     * When a user implements a gather flow that does not perform cache lookup as one cuda kernel, it might cause race condition with invalidate and commit operations.
     * This function marks the beginning of such a flow and tell the cache that all kernel launched between start_custom_flow and end_custom_flow are "atomic" with respect to invalidate and commit operations.
     * Note it users responsibility to ensure that any internal streams are supplied to EC event.
     * 
     * @return EC_SUCCESS on success, Might return other values based on implementation
     */
    virtual ECError start_custom_flow() = 0;

    /**
     * \brief End a custom flow
     * 
     * @return EC_SUCCESS on success, Might return other values based on implementation
     */
    virtual ECError end_custom_flow() = 0;

    /**
     * \brief Return the cache total capcity in Lines
    */
    virtual size_t get_max_num_embedding_vectors_in_cache() const = 0;

    virtual CacheAllocationSize get_lookup_context_size() const = 0;
    virtual CacheAllocationSize get_modify_context_size(uint32_t max_update_size) const = 0;

protected:
    CACHE_IMPLEMENTATION_TYPE type_;
    Allocator* allocator_;
    mutable Logger* logger_;
};
}
