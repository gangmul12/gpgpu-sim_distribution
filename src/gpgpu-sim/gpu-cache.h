// Copyright (c) 2009-2011, Tor M. Aamodt, Tayler Hetherington
// The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice, this
// list of conditions and the following disclaimer in the documentation and/or
// other materials provided with the distribution.
// Neither the name of The University of British Columbia nor the names of its
// contributors may be used to endorse or promote products derived from this
// software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef GPU_CACHE_H
#define GPU_CACHE_H

#include <stdio.h>
#include <stdlib.h>
#include "gpu-misc.h"
#include "mem_fetch.h"
#include "../abstract_hardware_model.h"
#include "../tr1_hash_map.h"

#include "addrdec.h"
#include <iostream>
#include "prefetcher.h"
#define MAX_DEFAULT_CACHE_SIZE_MULTIBLIER 4

enum cache_block_state {
    INVALID=0,
    RESERVED,
    VALID,
    MODIFIED,
	 PARTIAL
};

enum cache_request_status {
    HIT = 0,
    HIT_RESERVED,
    MISS,
    RESERVATION_FAIL, 
	SECTOR_MISS,
	PARTIAL_SECTOR_MISS,
    NUM_CACHE_REQUEST_STATUS
};

enum cache_reservation_fail_reason {
	LINE_ALLOC_FAIL= 0,// all line are reserved
	MISS_QUEUE_FULL,   // MISS queue (i.e. interconnect or DRAM) is full
	MSHR_ENRTY_FAIL,
	MSHR_MERGE_ENRTY_FAIL,
	MSHR_RW_PENDING,
    NUM_CACHE_RESERVATION_FAIL_STATUS
};

enum cache_event_type {
    WRITE_BACK_REQUEST_SENT,
    READ_REQUEST_SENT,
    WRITE_REQUEST_SENT,
	WRITE_ALLOCATE_SENT
};

struct evicted_block_info {
	new_addr_type m_block_addr;
	unsigned m_modified_size;
	evicted_block_info() {
		m_block_addr = 0;
		m_modified_size = 0;
	}
	void set_info(new_addr_type block_addr, unsigned modified_size){
		m_block_addr = block_addr;
		m_modified_size = modified_size;
	}
};

typedef std::bitset<32> mem_access_sector32_mask_t;
struct evicted_sector_block_info {
	new_addr_type m_block_addr;
	unsigned m_modified_size;
	mem_access_sector32_mask_t m_sector;
	evicted_sector_block_info(){
		m_block_addr=0;
		m_modified_size =0;
	}
	void set_info(new_addr_type block_addr, unsigned modified_size, mem_access_sector32_mask_t sector){
		m_block_addr = block_addr;
		m_modified_size = modified_size;
		m_sector = sector;
	}
};
struct cache_event {
	enum cache_event_type m_cache_event_type;
	evicted_block_info m_evicted_block; //if it was write_back event, fill the the evicted block info

	cache_event(enum cache_event_type m_cache_event){
		m_cache_event_type = m_cache_event;
	}

	cache_event(enum cache_event_type cache_event, evicted_block_info evicted_block){
	m_cache_event_type = cache_event;
	m_evicted_block = evicted_block;
	}
};

const char * cache_request_status_str(enum cache_request_status status); 

struct cache_block_t {
    cache_block_t()
    {
        m_tag=0;
        m_block_addr=0;
        m_prefetched=false;
        m_accessed = false;
        m_loaded_check = false;
    }

    virtual void allocate( new_addr_type tag, new_addr_type block_addr, unsigned time, mem_access_sector_mask_t sector_mask) = 0;
    virtual void fill( unsigned time, mem_access_sector_mask_t sector_mask) = 0;

    virtual bool is_invalid_line() = 0;
    virtual bool is_valid_line() = 0;
    virtual bool is_reserved_line() = 0;
    virtual bool is_modified_line() = 0;

    virtual enum cache_block_state get_status( mem_access_sector_mask_t sector_mask) = 0;
    virtual void set_status(enum cache_block_state m_status, mem_access_sector_mask_t sector_mask) = 0;

    virtual unsigned long long get_last_access_time() = 0;
    virtual void set_last_access_time(unsigned long long time, mem_access_sector_mask_t sector_mask) = 0;
    virtual unsigned long long get_alloc_time() = 0;
    virtual void set_ignore_on_fill(bool m_ignore, mem_access_sector_mask_t sector_mask) = 0;
    virtual void set_modified_on_fill(bool m_modified, mem_access_sector_mask_t sector_mask) = 0;
    virtual unsigned get_modified_size() = 0;
    virtual void set_m_readable(bool readable, mem_access_sector_mask_t sector_mask)=0;
    virtual bool is_readable(mem_access_sector_mask_t sector_mask)=0;
    virtual void print_status()=0;
    virtual ~cache_block_t() {}


    new_addr_type    m_tag;
    new_addr_type    m_block_addr;
    bool m_prefetched;
    bool m_accessed;
    bool m_loaded_check;

};

struct line_cache_block: public cache_block_t  {
	line_cache_block()
	    {
	        m_alloc_time=0;
	        m_fill_time=0;
	        m_last_access_time=0;
	        m_status=INVALID;
	        m_ignore_on_fill_status = false;
	        m_set_modified_on_fill = false;
	        m_readable = true;
	    }
	    void allocate( new_addr_type tag, new_addr_type block_addr, unsigned time, mem_access_sector_mask_t sector_mask)
	    {
	        m_tag=tag;
	        m_block_addr=block_addr;
	        m_alloc_time=time;
	        m_last_access_time=time;
	        m_fill_time=0;
	        m_status=RESERVED;
	        m_ignore_on_fill_status = false;
	        m_set_modified_on_fill = false;
        m_prefetched=false;
        m_accessed = false;
        m_loaded_check = false;
	    }
		void fill( unsigned time, mem_access_sector_mask_t sector_mask )
	    {
	    	//if(!m_ignore_on_fill_status)
	    	//	assert( m_status == RESERVED );

	    	m_status = m_set_modified_on_fill? MODIFIED : VALID;

	        m_fill_time=time;
	    }
		virtual bool is_invalid_line()
	    {
	    	return m_status == INVALID;
	    }
		virtual bool is_valid_line()
	    {
	    	 return m_status == VALID;
	    }
		virtual bool is_reserved_line()
	    {
	    	 return m_status == RESERVED;
	    }
		virtual bool is_modified_line()
	    {
	    	return m_status == MODIFIED;
	    }

		virtual enum cache_block_state get_status(mem_access_sector_mask_t sector_mask)
	    {
	    	return m_status;
	    }
		virtual void set_status(enum cache_block_state status, mem_access_sector_mask_t sector_mask)
	    {
	    	m_status = status;
	    }
		virtual unsigned long long get_last_access_time()
		{
			return m_last_access_time;
		}
		virtual void set_last_access_time(unsigned long long time, mem_access_sector_mask_t sector_mask)
	    {
	    	m_last_access_time = time;
	    }
		virtual unsigned long long get_alloc_time()
	    {
	    	return m_alloc_time;
	    }
		virtual void set_ignore_on_fill(bool m_ignore, mem_access_sector_mask_t sector_mask)
		{
			m_ignore_on_fill_status = m_ignore;
		}
		virtual void set_modified_on_fill(bool m_modified, mem_access_sector_mask_t sector_mask)
		{
	    	m_set_modified_on_fill = m_modified;
		}
		virtual unsigned  get_modified_size()
		{
			return SECTOR_CHUNCK_SIZE * SECTOR_SIZE;   //i.e. cache line size
		}
		virtual void set_m_readable(bool readable, mem_access_sector_mask_t sector_mask)
		{
			m_readable = readable;
		}
		virtual bool is_readable(mem_access_sector_mask_t sector_mask) {
			return m_readable;
		}
		virtual void print_status() {
			 printf("m_block_addr is %llu, status = %u\n", m_block_addr, m_status);
		}


private:
	    unsigned long long     m_alloc_time;
	    unsigned long long     m_last_access_time;
	    unsigned long long     m_fill_time;
	    cache_block_state    m_status;
	    bool m_ignore_on_fill_status;
	    bool m_set_modified_on_fill;
	    bool m_readable;
};

struct sector_cache_block : public cache_block_t {
	sector_cache_block()
    {
		init();
    }

	void init() {
		for(unsigned i =0; i< SECTOR_CHUNCK_SIZE; ++i) {
			m_sector_alloc_time[i]= 0;
			m_sector_fill_time[i]= 0;
			m_last_sector_access_time[i]= 0;
			m_status[i]= INVALID;
			m_ignore_on_fill_status[i] = false;
			m_set_modified_on_fill[i] = false;
			m_readable[i] = true;
			}
			m_line_alloc_time=0;
			m_line_last_access_time=0;
			m_line_fill_time=0;
	}

	virtual void allocate( new_addr_type tag, new_addr_type block_addr, unsigned time, mem_access_sector_mask_t sector_mask )
    {
    	allocate_line( tag,  block_addr,  time, sector_mask );
    }

    void allocate_line( new_addr_type tag, new_addr_type block_addr, unsigned time, mem_access_sector_mask_t sector_mask )
	{
		//allocate a new line
		//assert(m_block_addr != 0 && m_block_addr != block_addr);
		init();
		m_tag=tag;
		m_block_addr=block_addr;

		unsigned sidx = get_sector_index(sector_mask);

		//set sector stats
		m_sector_alloc_time[sidx]=time;
		m_last_sector_access_time[sidx]=time;
		m_sector_fill_time[sidx]=0;
		m_status[sidx]=RESERVED;
		m_ignore_on_fill_status[sidx] = false;
		m_set_modified_on_fill[sidx] = false;

		//set line stats
		m_line_alloc_time=time;   //only set this for the first allocated sector
		m_line_last_access_time=time;
		m_line_fill_time=0;
		m_prefetched=false;
		m_accessed = false;
      		m_loaded_check = false;
	}

    void allocate_sector(unsigned time, mem_access_sector_mask_t sector_mask )
	{
    	//allocate invalid sector of this allocated valid line
    	assert(is_valid_line());
		unsigned sidx = get_sector_index(sector_mask);

		//set sector stats
		m_sector_alloc_time[sidx]=time;
		m_last_sector_access_time[sidx]=time;
		m_sector_fill_time[sidx]=0;
		if(m_status[sidx]==MODIFIED)    //this should be the case only for fetch-on-write policy //TO DO
			m_set_modified_on_fill[sidx] = true;
		else
			m_set_modified_on_fill[sidx] = false;

		m_status[sidx]=RESERVED;
		m_ignore_on_fill_status[sidx] = false;
		//m_set_modified_on_fill[sidx] = false;
		m_readable[sidx] = true;

		//set line stats
		m_line_last_access_time=time;
		m_line_fill_time=0;
	}

    virtual void fill( unsigned time, mem_access_sector_mask_t sector_mask)
    {
    	unsigned sidx = get_sector_index(sector_mask);

    //	if(!m_ignore_on_fill_status[sidx])
    //	         assert( m_status[sidx] == RESERVED );

    	m_status[sidx] = m_set_modified_on_fill[sidx]? MODIFIED : VALID;

        m_sector_fill_time[sidx]=time;
        m_line_fill_time=time;
    }
    virtual bool is_invalid_line() {
    	//all the sectors should be invalid
    	for(unsigned i =0; i< SECTOR_CHUNCK_SIZE; ++i) {
    		if (m_status[i] != INVALID)
    			return false;
    	}
    	return true;
    }
    virtual bool is_valid_line() { return  !(is_invalid_line()); }
    virtual bool is_reserved_line() {
    	//if any of the sector is reserved, then the line is reserved
		for(unsigned i =0; i< SECTOR_CHUNCK_SIZE; ++i) {
			if (m_status[i] == RESERVED)
				return true;
		}
		return false;
    }
    virtual bool is_modified_line() {
    	//if any of the sector is modified, then the line is modified
    	for(unsigned i =0; i< SECTOR_CHUNCK_SIZE; ++i) {
			if (m_status[i] == MODIFIED)
				return true;
		}
		return false;
    }

    virtual enum cache_block_state get_status(mem_access_sector_mask_t sector_mask)
	{
    	unsigned sidx = get_sector_index(sector_mask);

		return m_status[sidx];
	}

    virtual void set_status(enum cache_block_state status, mem_access_sector_mask_t sector_mask)
	{
		unsigned sidx = get_sector_index(sector_mask);
		m_status[sidx] = status;
	}

    virtual unsigned long long get_last_access_time()
	{
		return m_line_last_access_time;
	}

    virtual void set_last_access_time(unsigned long long time, mem_access_sector_mask_t sector_mask)
	{
		unsigned sidx = get_sector_index(sector_mask);

		m_last_sector_access_time[sidx] = time;
		m_line_last_access_time = time;
	}

    virtual unsigned long long get_alloc_time()
	{
		return m_line_alloc_time;
	}

    virtual void set_ignore_on_fill(bool m_ignore, mem_access_sector_mask_t sector_mask)
	{
		unsigned sidx = get_sector_index(sector_mask);
		m_ignore_on_fill_status[sidx] = m_ignore;
	}

    virtual void set_modified_on_fill(bool m_modified, mem_access_sector_mask_t sector_mask)
	{
		unsigned sidx = get_sector_index(sector_mask);
		m_set_modified_on_fill[sidx] = m_modified;
	}

    virtual void set_m_readable(bool readable, mem_access_sector_mask_t sector_mask)
    {
    	unsigned sidx = get_sector_index(sector_mask);
    	m_readable[sidx] = readable;
    }

    virtual bool is_readable(mem_access_sector_mask_t sector_mask) {
    	unsigned sidx = get_sector_index(sector_mask);
    	return m_readable[sidx];
	}

    virtual unsigned  get_modified_size()
	{
		unsigned modified=0;
		for(unsigned i =0; i< SECTOR_CHUNCK_SIZE; ++i) {
			if (m_status[i] == MODIFIED)
				modified++;
		}
		return modified * SECTOR_SIZE;
	}

    virtual void print_status() {
    	 printf("m_block_addr is %llu, status = %u %u %u %u\n", m_block_addr, m_status[0], m_status[1], m_status[2], m_status[3]);
    }


private:
    unsigned m_sector_alloc_time[SECTOR_CHUNCK_SIZE];
    unsigned m_last_sector_access_time[SECTOR_CHUNCK_SIZE];
    unsigned m_sector_fill_time[SECTOR_CHUNCK_SIZE];
    unsigned m_line_alloc_time;
    unsigned m_line_last_access_time;
    unsigned m_line_fill_time;
    cache_block_state    m_status[SECTOR_CHUNCK_SIZE];
    bool m_ignore_on_fill_status[SECTOR_CHUNCK_SIZE];
    bool m_set_modified_on_fill[SECTOR_CHUNCK_SIZE];
    bool m_readable[SECTOR_CHUNCK_SIZE];

    unsigned get_sector_index(mem_access_sector_mask_t sector_mask)
    {
    	assert(sector_mask.count() == 1);
    	for(unsigned i =0; i< SECTOR_CHUNCK_SIZE; ++i) {
    		if(sector_mask.to_ulong() & (1<<i))
    			return i;
    	}
    }
};
const unsigned SECTOR_CHUNK_SIZE32=32;

struct buffer_block_t {
    buffer_block_t()
    {
        m_tag=0;
        m_block_addr=0;
        m_accessed = false;
    }

    virtual void allocate( new_addr_type tag, new_addr_type block_addr, unsigned time, mem_access_sector32_mask_t sector_mask) = 0;

    virtual bool is_invalid_line() = 0;
    virtual bool is_valid_line() = 0;
    virtual bool is_reserved_line() = 0;
    virtual bool is_modified_line() = 0;

    virtual enum cache_block_state get_status( mem_access_sector32_mask_t sector_mask) = 0;
    virtual void set_status(enum cache_block_state m_status, mem_access_sector32_mask_t sector_mask) = 0;

    virtual unsigned long long get_last_access_time() = 0;
    virtual void set_last_access_time(unsigned long long time, mem_access_sector32_mask_t sector_mask) = 0;
    virtual unsigned long long get_alloc_time() = 0;
    virtual unsigned get_modified_size() = 0;
    virtual void print_status()=0;
    virtual ~buffer_block_t() {}


    new_addr_type    m_tag;
    new_addr_type    m_block_addr;
    bool m_accessed;

};


struct sector32_cache_block : public buffer_block_t {
	sector32_cache_block()
    {
		init();
    }

	void init() {
		for(unsigned i =0; i< SECTOR_CHUNK_SIZE32; ++i) {
			m_sector_alloc_time[i]= 0;
			m_sector_fill_time[i]= 0;
			m_last_sector_access_time[i]= 0;
			m_status[i]= INVALID;
			}
			m_line_alloc_time=0;
			m_line_last_access_time=0;
			m_line_fill_time=0;
	}

	virtual void allocate( new_addr_type tag, new_addr_type block_addr, unsigned time, mem_access_sector32_mask_t sector_mask )
    {
    	allocate_line( tag,  block_addr,  time, sector_mask );
    }

    void allocate_line( new_addr_type tag, new_addr_type block_addr, unsigned time, mem_access_sector32_mask_t sector_mask )
	{
		//allocate a new line
		//assert(m_block_addr != 0 && m_block_addr != block_addr);
		init();
		m_tag=tag;
		m_block_addr=block_addr;

		//set sector stats
		for(unsigned i = 0; i < SECTOR_CHUNK_SIZE32; i++){
			if(sector_mask[i]){
				m_sector_alloc_time[i] = time;
				m_last_sector_access_time[i] = time;
				m_sector_fill_time[i] = time;
				m_status[i] = MODIFIED;
			}
		}
		//set line stats
		m_line_last_access_time=time;
		m_line_fill_time=time;

		//set line stats
		m_line_alloc_time=time;   //only set this for the first allocated sector
		m_accessed = false;
	}

    void allocate_sector(unsigned time, mem_access_sector32_mask_t sector_mask )
	{
    	//allocate invalid sector of this allocated valid line
    	assert(is_valid_line());

		//set sector stats
		for(unsigned i = 0; i < SECTOR_CHUNK_SIZE32; i++){
			if(sector_mask[i]){
				m_sector_alloc_time[i] = time;
				m_last_sector_access_time[i] = time;
				m_sector_fill_time[i] = time;
				m_status[i] = MODIFIED;
			}
		}
		//set line stats
		m_line_last_access_time=time;
		m_line_fill_time=time;
	}
	virtual bool is_valid_line(){return !is_invalid_line();}
    virtual bool is_invalid_line() {
    	//all the sectors should be invalid
    	for(unsigned i =0; i< SECTOR_CHUNK_SIZE32; ++i) {
    		if (m_status[i] != INVALID)
    			return false;
    	}
    	return true;
    }
    virtual bool is_reserved_line() {
    	//if any of the sector is reserved, then the line is reserved
		for(unsigned i =0; i< SECTOR_CHUNK_SIZE32; ++i) {
			if (m_status[i] == RESERVED)
				return true;
		}
		return false;
    }
    virtual bool is_modified_line() {
    	//if any of the sector is modified, then the line is modified
    	for(unsigned i =0; i< SECTOR_CHUNK_SIZE32; ++i) {
			if (m_status[i] == MODIFIED)
				return true;
		}
		return false;
    }

    virtual enum cache_block_state get_status(mem_access_sector32_mask_t sector_mask)
	{
		enum cache_block_state result;
		bool is_first=true;
		for(unsigned i = 0 ; i < SECTOR_CHUNK_SIZE32; ++i){
			if(sector_mask[i]==true){
				if(is_first){
					result = m_status[i];
					is_first=false;
				}
				else{
					assert(m_status[i]!=RESERVED);
					assert(m_status[i]!=VALID);
					if(result != m_status[i]){
						result = PARTIAL; 
					}
				}
			}
		}
		assert(!is_first);
		return result;
	}

    virtual void set_status(enum cache_block_state status, mem_access_sector32_mask_t sector_mask)
	{
		for(unsigned i = 0 ; i < SECTOR_CHUNK_SIZE32 ; i++){
			if(sector_mask[i]){
				m_status[i] = status;
			}
		}

	}

    virtual unsigned long long get_last_access_time()
	{
		return m_line_last_access_time;
	}

    virtual void set_last_access_time(unsigned long long time, mem_access_sector32_mask_t sector_mask)
	{
		for(unsigned i = 0 ; i < SECTOR_CHUNK_SIZE32 ; i++){
			if(sector_mask[i]){
				m_last_sector_access_time[i] = time;
			}
		}
		m_line_last_access_time = time;
	}

    virtual unsigned long long get_alloc_time()
	{
		return m_line_alloc_time;
	}


    virtual unsigned  get_modified_size()
	{
		unsigned modified=0;
		for(unsigned i =0; i< SECTOR_CHUNK_SIZE32; ++i) {
			if (m_status[i] == MODIFIED)
				modified++;
		}
		return modified * SECTOR_SIZE;
	}

    virtual void print_status() {
    	 printf("m_block_addr is %llu, status = %u %u %u %u\n", m_block_addr, m_status[0], m_status[1], m_status[2], m_status[3]);
    }

	 void invalidate(){
		for(unsigned i = 0 ; i < SECTOR_CHUNK_SIZE32 ; i++)
				m_status[i] = INVALID;
	 }
	 mem_access_sector32_mask_t get_dirty_mask(){
	 	mem_access_sector32_mask_t result;
		for(unsigned i = 0 ; i < SECTOR_CHUNK_SIZE32 ; i++){
			if(m_status[i]==MODIFIED)
				result.set(i);
		}
		return result;
	 }


private:
    unsigned m_sector_alloc_time[SECTOR_CHUNK_SIZE32];
    unsigned m_last_sector_access_time[SECTOR_CHUNK_SIZE32];
    unsigned m_sector_fill_time[SECTOR_CHUNK_SIZE32];
    unsigned m_line_alloc_time;
    unsigned m_line_last_access_time;
    unsigned m_line_fill_time;
    cache_block_state    m_status[SECTOR_CHUNK_SIZE32];

    unsigned get_sector_index(mem_access_sector32_mask_t sector_mask)
    {
    	assert(sector_mask.count() == 1);
    	for(unsigned i =0; i< SECTOR_CHUNK_SIZE32; ++i) {
    		if(sector_mask.to_ulong() & (1<<i))
    			return i;
    	}
    }
};

enum replacement_policy_t {
    LRU,
    FIFO
};

enum write_policy_t {
    READ_ONLY,
    WRITE_BACK,
    WRITE_THROUGH,
    WRITE_EVICT,
    LOCAL_WB_GLOBAL_WT
};

enum allocation_policy_t {
    ON_MISS,
    ON_FILL,
	STREAMING
};


enum write_allocate_policy_t {
	NO_WRITE_ALLOCATE,
	WRITE_ALLOCATE,
	FETCH_ON_WRITE,
	LAZY_FETCH_ON_READ
};

enum mshr_config_t {
    TEX_FIFO, // Tex cache
    ASSOC, // normal cache
	SECTOR_TEX_FIFO,  //Tex cache sends requests to high-level sector cache
	SECTOR_ASSOC // normal cache sends requests to high-level sector cache
};

enum set_index_function{
	LINEAR_SET_FUNCTION = 0,
	BITWISE_XORING_FUNCTION,
	HASH_IPOLY_FUNCTION,
	FERMI_HASH_SET_FUNCTION,
    CUSTOM_SET_FUNCTION
};

enum cache_type{
    NORMAL = 0,
    SECTOR,
    LARGE
};

#define MAX_WARP_PER_SHADER 64
#define INCT_TOTAL_BUFFER 64
#define L2_TOTAL 64
#define MAX_WARP_PER_SHADER 64
#define MAX_WARP_PER_SHADER 64

class cache_config {
public:
    cache_config() 
    { 
        m_valid = false; 
        m_disabled = false;
        m_config_string = NULL; // set by option parser
        m_config_stringPrefL1 = NULL;
        m_config_stringPrefShared = NULL;
        m_data_port_width = 0;
        m_set_index_function = LINEAR_SET_FUNCTION;
        m_is_streaming = false;
	have_prefetcher = false;
	m_bypass_global = false;
	m_fill_port_width = 0;
	m_has_write_buffer=false;
    }
    void init(char * config, FuncCache status)
    {
    	cache_status= status;
        assert( config );
        char ct, rp, wp, ap, mshr_type, wap, sif;

        int ntok = sscanf(config,"%c:%u:%u:%u,%c:%c:%c:%c:%c,%c:%u:%u,%u:%u,%u:%u",
                          &ct, &m_nset, &m_line_sz, &m_assoc, &rp, &wp, &ap, &wap,
                          &sif,&mshr_type,&m_mshr_entries,&m_mshr_max_merge,
                          &m_miss_queue_size, &m_result_fifo_entries,
                          &m_data_port_width, &m_fill_port_width);
        if ( ntok < 12 ) {
            if ( !strcmp(config,"none") ) {
                m_disabled = true;
                return;
            }
            exit_parse_error();
        }

        switch (ct) {
			   case 'N': m_cache_type = NORMAL; break;
			   case 'S': m_cache_type = SECTOR; break;
			   case 'L': m_cache_type = LARGE; break;
			   default: exit_parse_error();
        }
	if(m_cache_type == LARGE) assert(mshr_type=='S');
        switch (rp) {
               case 'L': m_replacement_policy = LRU; break;
               case 'F': m_replacement_policy = FIFO; break;
               default: exit_parse_error();
        }
        switch (rp) {
        case 'L': m_replacement_policy = LRU; break;
        case 'F': m_replacement_policy = FIFO; break;
        default: exit_parse_error();
        }
        switch (wp) {
        case 'R': m_write_policy = READ_ONLY; break;
        case 'B': m_write_policy = WRITE_BACK; break;
        case 'T': m_write_policy = WRITE_THROUGH; break;
        case 'E': m_write_policy = WRITE_EVICT; break;
        case 'L': m_write_policy = LOCAL_WB_GLOBAL_WT; break;
        default: exit_parse_error();
        }
        switch (ap) {
        case 'm': m_alloc_policy = ON_MISS; break;
        case 'f': m_alloc_policy = ON_FILL; break;
        case 's': m_alloc_policy = STREAMING; break;
        default: exit_parse_error();
        }
        if(m_alloc_policy == STREAMING) {
        	//For streaming cache, we set the alloc policy to be on-fill to remove all line_alloc_fail stalls
        	//we set the MSHRs to be equal to max allocated cache lines. This is possible by moving TAG to be shared between cache line and MSHR enrty (i.e. for each cache line, there is an MSHR rntey associated with it)
        	//This is the easiest think we can think about to model (mimic) L1 streaming cache in Pascal and Volta
        	//Based on our microbenchmakrs, MSHRs entries have been increasing substantially in Pascal and Volta
        	//For more information about streaming cache, see:
        	// http://on-demand.gputechconf.com/gtc/2017/presentation/s7798-luke-durant-inside-volta.pdf
        	// https://ieeexplore.ieee.org/document/8344474/
        	m_is_streaming = true;
			m_alloc_policy = ON_FILL;
			m_mshr_entries = m_nset*m_assoc*MAX_DEFAULT_CACHE_SIZE_MULTIBLIER;
			if(m_cache_type == SECTOR)
				m_mshr_entries *=  SECTOR_CHUNCK_SIZE;
			m_mshr_max_merge = MAX_WARP_PER_SM;
        }
        switch (mshr_type) {
        case 'F': m_mshr_type = TEX_FIFO; assert(ntok==14); break;
        case 'T': m_mshr_type = SECTOR_TEX_FIFO; assert(ntok==14); break;
        case 'A': m_mshr_type = ASSOC; break;
        case 'S' : m_mshr_type = SECTOR_ASSOC; break;
        default: exit_parse_error();
        }
        m_line_sz_log2 = LOGB2(m_line_sz);
        m_nset_log2 = LOGB2(m_nset);
        m_valid = true;
        m_atom_sz = (m_cache_type == SECTOR)? SECTOR_SIZE : m_line_sz;
        original_m_assoc = m_assoc;

        //For more details about difference between FETCH_ON_WRITE and WRITE VALIDAE policies
        //Read: Jouppi, Norman P. "Cache write policies and performance". ISCA 93.
        //WRITE_ALLOCATE is the old write policy in GPGPU-sim 3.x, that send WRITE and READ for every write request
        switch(wap){
        case 'N': m_write_alloc_policy = NO_WRITE_ALLOCATE; break;
        case 'W': m_write_alloc_policy = WRITE_ALLOCATE; break;
        case 'F': m_write_alloc_policy = FETCH_ON_WRITE; break;
        case 'L': m_write_alloc_policy = LAZY_FETCH_ON_READ; break;
		default: exit_parse_error();
        }

        // detect invalid configuration 
        if (m_alloc_policy == ON_FILL and m_write_policy == WRITE_BACK) {
            // A writeback cache with allocate-on-fill policy will inevitably lead to deadlock:  
            // The deadlock happens when an incoming cache-fill evicts a dirty
            // line, generating a writeback request.  If the memory subsystem
            // is congested, the interconnection network may not have
            // sufficient buffer for the writeback request.  This stalls the
            // incoming cache-fill.  The stall may propagate through the memory
            // subsystem back to the output port of the same core, creating a
            // deadlock where the wrtieback request and the incoming cache-fill
            // are stalling each other.  
            assert(0 && "Invalid cache configuration: Writeback cache cannot allocate new line on fill. "); 
        }

        if((m_write_alloc_policy == FETCH_ON_WRITE || m_write_alloc_policy == LAZY_FETCH_ON_READ )&& m_alloc_policy == ON_FILL)
		{
			assert(0 && "Invalid cache configuration: FETCH_ON_WRITE and LAZY_FETCH_ON_READ cannot work properly with ON_FILL policy. Cache must be ON_MISS. ");
		}
        if(m_cache_type == SECTOR)
		{
			assert(m_line_sz / SECTOR_SIZE == SECTOR_CHUNCK_SIZE && m_line_sz % SECTOR_SIZE == 0);
		}

        // default: port to data array width and granularity = line size 
        if (m_data_port_width == 0) {
            m_data_port_width = m_line_sz; 
        }
        assert(m_line_sz % m_data_port_width == 0); 
	if (m_fill_port_width == 0){
	    m_fill_port_width = m_data_port_width;
	}
	assert(m_line_sz % m_fill_port_width == 0);
        switch(sif){
        case 'H': m_set_index_function = FERMI_HASH_SET_FUNCTION; break;
        case 'P': m_set_index_function = HASH_IPOLY_FUNCTION; break;
        case 'C': m_set_index_function = CUSTOM_SET_FUNCTION; break;
        case 'L': m_set_index_function = LINEAR_SET_FUNCTION; break;
        default: exit_parse_error();
        }
    }
    bool disabled() const { return m_disabled;}
    unsigned get_line_sz() const
    {
        assert( m_valid );
        return m_line_sz;
    }
    unsigned get_atom_sz() const
	{
		assert( m_valid );
		return m_atom_sz;
	}
    unsigned get_num_lines() const
    {
        assert( m_valid );
        return m_nset * m_assoc;
    }
    unsigned get_max_num_lines() const
    {
        assert( m_valid );
        return MAX_DEFAULT_CACHE_SIZE_MULTIBLIER * m_nset * original_m_assoc;
    }
    void print( FILE *fp ) const
    {
        fprintf( fp, "Size = %d B (%d Set x %d-way x %d byte line)\n", 
                 m_line_sz * m_nset * m_assoc,
                 m_nset, m_assoc, m_line_sz );
    }

    virtual unsigned set_index( new_addr_type addr ) const
    {
        if(m_set_index_function != LINEAR_SET_FUNCTION){
            printf("\nGPGPU-Sim cache configuration error: Hashing or "
                    "custom set index function selected in configuration "
                    "file for a cache that has not overloaded the set_index "
                    "function\n");
            abort();
        }
        return(addr >> m_line_sz_log2) & (m_nset-1);
    }

    new_addr_type tag( new_addr_type addr ) const
    {
        // For generality, the tag includes both index and tag. This allows for more complex set index
        // calculations that can result in different indexes mapping to the same set, thus the full
        // tag + index is required to check for hit/miss. Tag is now identical to the block address.

        //return addr >> (m_line_sz_log2+m_nset_log2);
        return addr & ~(new_addr_type)(m_line_sz-1);
    }
    new_addr_type block_addr( new_addr_type addr ) const
    {
        return addr & ~(new_addr_type)(m_line_sz-1);
    }
    new_addr_type mshr_addr( new_addr_type addr ) const
	{
    	return addr & ~(new_addr_type)(m_atom_sz-1);
	}
    enum mshr_config_t get_mshr_type() const
	{
    	return m_mshr_type;
	}
    void set_assoc(unsigned n)
	{
    	//set new assoc. L1 cache dynamically resized in Volta
    	m_assoc = n;
	}
    unsigned get_nset() const
	{
		assert( m_valid );
		return m_nset;
	}
    unsigned get_total_size_inKB() const
	{
		assert( m_valid );
		return (m_assoc*m_nset*m_line_sz)/1024;
	}
    bool is_streaming() {
    	return m_is_streaming;
    }
    FuncCache get_cache_status() {return cache_status;}
    char *m_config_string;
    char *m_config_stringPrefL1;
    char *m_config_stringPrefShared;
    FuncCache cache_status;
    bool have_prefetcher;
	 bool m_bypass_global;
	 unsigned fetch_mask;
    enum cache_type m_cache_type;

	 bool m_has_write_buffer;
protected:
    void exit_parse_error()
    {
        printf("GPGPU-Sim uArch: cache configuration parsing error (%s)\n", m_config_string );
        abort();
    }

    bool m_valid;
    bool m_disabled;
    unsigned m_line_sz;
    unsigned m_line_sz_log2;
    unsigned m_nset;
    unsigned m_nset_log2;
    unsigned m_assoc;
    unsigned m_atom_sz;
    unsigned original_m_assoc;
    bool m_is_streaming;

    enum replacement_policy_t m_replacement_policy; // 'L' = LRU, 'F' = FIFO
    enum write_policy_t m_write_policy;             // 'T' = write through, 'B' = write back, 'R' = read only
    enum allocation_policy_t m_alloc_policy;        // 'm' = allocate on miss, 'f' = allocate on fill
    enum mshr_config_t m_mshr_type;

    write_allocate_policy_t m_write_alloc_policy;	// 'W' = Write allocate, 'N' = No write allocate

    union {
        unsigned m_mshr_entries;
        unsigned m_fragment_fifo_entries;
    };
    union {
        unsigned m_mshr_max_merge;
        unsigned m_request_fifo_entries;
    };
    union {
        unsigned m_miss_queue_size;
        unsigned m_rob_entries;
    };
    unsigned m_result_fifo_entries;
    unsigned m_data_port_width; //< number of byte the cache can access per cycle 
    unsigned m_fill_port_width;
    enum set_index_function m_set_index_function; // Hash, linear, or custom set index function
    friend class tag_array;
    friend class baseline_cache;
    friend class read_only_cache;
    friend class tex_cache;
    friend class data_cache;
    friend class l1_cache;
    friend class l2_cache;
    friend class memory_sub_partition;
};

class l1d_cache_config : public cache_config{
public:
	l1d_cache_config() : cache_config(){}
	virtual unsigned set_index(new_addr_type addr) const;
	unsigned l1_latency;
};

class l2_cache_config : public cache_config {
public:
	l2_cache_config() : cache_config(){}
	void init(linear_to_raw_address_translation *address_mapping);
	virtual unsigned set_index(new_addr_type addr) const;

private:
	linear_to_raw_address_translation *m_address_mapping;
};

class tag_array {
public:
    // Use this constructor
    tag_array(cache_config &config, int core_id, int type_id );
    ~tag_array();

    enum cache_request_status probe( new_addr_type addr, unsigned &idx, mem_fetch* mf, bool probe_mode=false ) const;
    enum cache_request_status probe( new_addr_type addr, unsigned &idx, mem_access_sector_mask_t mask, bool probe_mode=false, mem_fetch* mf = NULL ) const;
    enum cache_request_status access( new_addr_type addr, unsigned time, unsigned &idx, mem_fetch* mf );
    enum cache_request_status access( new_addr_type addr, unsigned time, unsigned &idx, bool &wb, evicted_block_info &evicted, mem_fetch* mf );

    void fill( new_addr_type addr, unsigned time, mem_fetch* mf );
    void fill( unsigned idx, unsigned time, mem_fetch* mf );
    void fill( new_addr_type addr, unsigned time, mem_access_sector_mask_t mask );
    void fill( new_addr_type addr, unsigned time, mem_access_sector_mask_t mask, bool is_prefetched );

    unsigned size() const { return m_config.get_num_lines();}
    cache_block_t* get_block(unsigned idx) { return m_lines[idx];}

    void flush(); // flush all written entries
    void invalidate(); // invalidate all entries
    void new_window();

    void print( FILE *stream, unsigned &total_access, unsigned &total_misses ) const;
    float windowed_miss_rate( ) const;
    void get_stats(unsigned &total_access, unsigned &total_misses, unsigned &total_hit_res, unsigned &total_res_fail) const;

	void update_cache_parameters(cache_config &config);
	void add_pending_line(mem_fetch *mf);
	void remove_pending_line(mem_fetch *mf);
protected:
    // This constructor is intended for use only from derived classes that wish to
    // avoid unnecessary memory allocation that takes place in the
    // other tag_array constructor
    tag_array( cache_config &config,
               int core_id,
               int type_id,
               cache_block_t** new_lines );
    void init( int core_id, int type_id );

protected:

    cache_config &m_config;

    cache_block_t **m_lines; /* nbanks x nset x assoc lines in total */

    unsigned m_access;
    unsigned m_miss;
    unsigned m_pending_hit; // number of cache miss that hit a line that is allocated but not filled
    unsigned m_res_fail;
    unsigned m_sector_miss;

    // performance counters for calculating the amount of misses within a time window
    unsigned m_prev_snapshot_access;
    unsigned m_prev_snapshot_miss;
    unsigned m_prev_snapshot_pending_hit;

    int m_core_id; // which shader core is using this
    int m_type_id; // what kind of cache is this (normal, texture, constant)

    bool is_used;  //a flag if the whole cache has ever been accessed before

    typedef tr1_hash_map<new_addr_type,unsigned> line_table;
    line_table pending_lines;
};

class buffer_tag_array {
public:
    // Use this constructor
    buffer_tag_array(int core_id, int type_id );
    ~buffer_tag_array();

    enum cache_request_status probe( new_addr_type addr, unsigned &idx, mem_fetch* mf) const;
    enum cache_request_status access( new_addr_type addr, unsigned time, unsigned &idx, bool &wb, evicted_sector_block_info &evicted, mem_fetch* mf );

    unsigned size() const { return m_line_num;}
    sector32_cache_block* get_block(unsigned idx) { return m_lines[idx];}

    void flush(); // flush all written entries
    void invalidate(); // invalidate all entries


protected:

	 friend class write_buffer;
    sector32_cache_block **m_lines; /* nbanks x nset x assoc lines in total */

    unsigned m_access;
    unsigned m_miss;
    unsigned m_pending_hit; // number of cache miss that hit a line that is allocated but not filled
    unsigned m_res_fail;
    unsigned m_sector_miss;


    int m_core_id; // which shader core is using this
    int m_type_id; // what kind of cache is this (normal, texture, constant)
	 unsigned m_line_num;
	 unsigned m_line_size_log2;
	 unsigned m_line_size;
	 unsigned get_set_index(new_addr_type addr)const {return (addr >> m_line_size_log2)&(m_line_num-1);}
	 new_addr_type get_tag(new_addr_type addr)const {return addr & ~(new_addr_type)(m_line_size-1);}
	 new_addr_type get_block_addr(new_addr_type addr)const {return get_tag(addr);}
	 void calculate_mask(mem_fetch* mf, mem_access_sector32_mask_t &mask) const;



};


class mshr_table {
public:
    mshr_table( unsigned num_entries, unsigned max_merged)
    : m_num_entries(num_entries),
    m_max_merged(max_merged)
#if (tr1_hash_map_ismap == 0)
    ,m_data(2*num_entries)
#endif
    {
	m_have_prefetcher = false;
    }

    bool address_list_is_empty(new_addr_type block_addr){
        table::const_iterator i=m_data.find(block_addr);
        assert(i != m_data.end());

        return i->second.m_list.empty();
    }
    void set_prefetch(){m_have_prefetcher=true;}

    /// Checks if there is a pending request to the lower memory level already
    bool probe( new_addr_type block_addr ) const;
    /// Checks if there is space for tracking a new memory access
    bool full( new_addr_type block_addr ) const;
    /// Add or merge this access
    void add( new_addr_type block_addr, mem_fetch *mf );
    /// Returns true if cannot accept new fill responses
    bool busy() const {return false;}
    /// Accept a new cache fill response: mark entry ready for processing
    void mark_ready( new_addr_type block_addr, bool &has_atomic );
    void mark_ready_prefetch( new_addr_type block_addr, bool &has_atomic );
    void delete_content( new_addr_type block_addr );
    bool mshr_content_is_prefetched(mem_fetch* mf){
        return mf->is_prefetched();
    }
    void delete_prefetched_req_in_mshr( new_addr_type block_addr );

    /// Returns true if ready accesses exist
    bool access_ready() const {return !m_current_response.empty();}
    /// Returns next ready access
    mem_fetch *next_access();
    void display( FILE *fp ) const;
    // Returns true if there is a pending read after write
    bool is_read_after_write_pending(new_addr_type block_addr);

    void check_mshr_parameters( unsigned num_entries, unsigned max_merged )
    {
    	assert(m_num_entries==num_entries && "Change of MSHR parameters between kernels is not allowed");
    	assert(m_max_merged==max_merged && "Change of MSHR parameters between kernels is not allowed");
    }

private:

    // finite sized, fully associative table, with a finite maximum number of merged requests
    const unsigned m_num_entries;
    const unsigned m_max_merged;

    struct mshr_entry {
        std::list<mem_fetch*> m_list;
        bool m_has_atomic; 
        mshr_entry() : m_has_atomic(false) { }
    }; 
    typedef tr1_hash_map<new_addr_type,mshr_entry> table;
    typedef tr1_hash_map<new_addr_type,mshr_entry> line_table;
    table m_data;
    line_table pending_lines;

    // it may take several cycles to process the merged requests
    bool m_current_response_ready;
    std::list<new_addr_type> m_current_response;

	 bool m_have_prefetcher;

};


/***************************************************************** Caches *****************************************************************/
///
/// Simple struct to maintain cache accesses, misses, pending hits, and reservation fails.
///
struct cache_sub_stats{
    unsigned long long accesses;
    unsigned long long misses;
    unsigned long long pending_hits;
    unsigned long long res_fails;

    unsigned long long port_available_cycles; 
    unsigned long long data_port_busy_cycles; 
    unsigned long long fill_port_busy_cycles; 
    unsigned prefetch_stall_cycle;
    unsigned n_prefetch_access;
    unsigned n_prefetch_hit;
    unsigned n_issued_prefetch;
    unsigned n_prefetch_distance;
    unsigned n_prefetch_reserved_hit;
    unsigned n_prefetch_accessed_once;
    unsigned n_late_distance;

    cache_sub_stats(){
        clear();
    }
    void clear(){
        accesses = 0;
        misses = 0;
        pending_hits = 0;
        res_fails = 0;
        port_available_cycles = 0; 
        data_port_busy_cycles = 0; 
        fill_port_busy_cycles = 0;
        prefetch_stall_cycle =0;
        n_prefetch_access=0;
        n_prefetch_hit=0;
        n_issued_prefetch=0;
        n_prefetch_distance=0;
        n_prefetch_reserved_hit=0;
        n_prefetch_accessed_once=0;
        n_late_distance=0;
    }
    cache_sub_stats &operator+=(const cache_sub_stats &css){
        ///
        /// Overloading += operator to easily accumulate stats
        ///
        accesses += css.accesses;
        misses += css.misses;
        pending_hits += css.pending_hits;
        res_fails += css.res_fails;
        port_available_cycles += css.port_available_cycles; 
        data_port_busy_cycles += css.data_port_busy_cycles; 
        fill_port_busy_cycles += css.fill_port_busy_cycles; 
        prefetch_stall_cycle += css.prefetch_stall_cycle;
        n_prefetch_access += css.n_prefetch_access;
        n_prefetch_hit += css.n_prefetch_hit;
        n_issued_prefetch += css.n_issued_prefetch;
        n_prefetch_distance += css.n_prefetch_distance;
        n_prefetch_reserved_hit += css.n_prefetch_reserved_hit;
        n_prefetch_accessed_once += css.n_prefetch_accessed_once;
        n_late_distance += css.n_late_distance;
	return *this;
    }

    cache_sub_stats operator+(const cache_sub_stats &cs){
        ///
        /// Overloading + operator to easily accumulate stats
        ///
        cache_sub_stats ret;
        ret.accesses = accesses + cs.accesses;
        ret.misses = misses + cs.misses;
        ret.pending_hits = pending_hits + cs.pending_hits;
        ret.res_fails = res_fails + cs.res_fails;
        ret.port_available_cycles = port_available_cycles + cs.port_available_cycles; 
        ret.data_port_busy_cycles = data_port_busy_cycles + cs.data_port_busy_cycles; 
        ret.fill_port_busy_cycles = fill_port_busy_cycles + cs.fill_port_busy_cycles; 
	ret.prefetch_stall_cycle = prefetch_stall_cycle + cs.prefetch_stall_cycle;
        ret.n_prefetch_access = n_prefetch_access + cs.n_prefetch_access;
        ret.n_prefetch_hit = n_prefetch_hit + cs.n_prefetch_hit;
        ret.n_issued_prefetch = n_issued_prefetch + cs.n_issued_prefetch;
        ret.n_prefetch_distance = n_prefetch_distance + cs.n_prefetch_distance;
        ret.n_prefetch_reserved_hit = n_prefetch_reserved_hit+ cs.n_prefetch_reserved_hit;
        ret.n_prefetch_accessed_once = n_prefetch_accessed_once+ cs.n_prefetch_accessed_once;
        ret.n_late_distance = n_late_distance + cs.n_late_distance;
	return ret;
    }

    void print_port_stats(FILE *fout, const char *cache_name) const; 
};


// Used for collecting AerialVision per-window statistics
struct cache_sub_stats_pw{
    unsigned accesses;
    unsigned write_misses;
    unsigned write_hits;
    unsigned write_pending_hits;
    unsigned write_res_fails;

    unsigned read_misses;
    unsigned read_hits;
    unsigned read_pending_hits;
    unsigned read_res_fails;

    cache_sub_stats_pw(){
        clear();
    }
    void clear(){
        accesses = 0;
        write_misses = 0;
        write_hits = 0;
        write_pending_hits = 0;
        write_res_fails = 0;
        read_misses = 0;
        read_hits = 0;
        read_pending_hits = 0;
        read_res_fails = 0;
    }
    cache_sub_stats_pw &operator+=(const cache_sub_stats_pw &css){
        ///
        /// Overloading += operator to easily accumulate stats
        ///
        accesses += css.accesses;
        write_misses += css.write_misses;
        read_misses += css.read_misses;
        write_pending_hits += css.write_pending_hits;
        read_pending_hits += css.read_pending_hits;
        write_res_fails += css.write_res_fails;
        read_res_fails += css.read_res_fails;
        return *this;
    }

    cache_sub_stats_pw operator+(const cache_sub_stats_pw &cs){
        ///
        /// Overloading + operator to easily accumulate stats
        ///
        cache_sub_stats_pw ret;
        ret.accesses = accesses + cs.accesses;
        ret.write_misses = write_misses + cs.write_misses;
        ret.read_misses = read_misses + cs.read_misses;
        ret.write_pending_hits = write_pending_hits + cs.write_pending_hits;
        ret.read_pending_hits = read_pending_hits + cs.read_pending_hits;
        ret.write_res_fails = write_res_fails + cs.write_res_fails;
        ret.read_res_fails = read_res_fails + cs.read_res_fails;
        return ret;
    }

};


///
/// Cache_stats
/// Used to record statistics for each cache.
/// Maintains a record of every 'mem_access_type' and its resulting
/// 'cache_request_status' : [mem_access_type][cache_request_status]
///
class cache_stats {
public:
    cache_stats();
    void clear();
    // Clear AerialVision cache stats after each window
    void clear_pw();
    void inc_stats(int access_type, int access_outcome);
    // Increment AerialVision cache stats
    void inc_stats_pw(int access_type, int access_outcome);
    void inc_fail_stats(int access_type, int fail_outcome);
    enum cache_request_status select_stats_status(enum cache_request_status probe, enum cache_request_status access) const;
    unsigned long long &operator()(int access_type, int access_outcome, bool fail_outcome);
    unsigned long long operator()(int access_type, int access_outcome, bool fail_outcome) const;
    cache_stats operator+(const cache_stats &cs);
    cache_stats &operator+=(const cache_stats &cs);
    void print_stats(FILE *fout, const char *cache_name = "Cache_stats") const;
    void print_fail_stats(FILE *fout, const char *cache_name = "Cache_fail_stats") const;

    unsigned long long get_stats(enum mem_access_type *access_type, unsigned num_access_type, enum cache_request_status *access_status, unsigned num_access_status)  const;
    void get_sub_stats(struct cache_sub_stats &css) const;

    // Get per-window cache stats for AerialVision
    void get_sub_stats_pw(struct cache_sub_stats_pw &css) const;

    void sample_cache_port_utility(bool data_port_busy, bool fill_port_busy);
    void inc_num_prefetched(){ m_n_prefetch_access++;};
    void inc_num_prefetch_hit(){ m_n_prefetch_hit++; };
    void inc_prefetch_stall_cycle(unsigned time){m_prefetch_stall_cycle += time;};
    void inc_issued_prefetch(){ m_n_issued_prefetch++; };
    void inc_prefetch_distance(unsigned time){ m_n_prefetch_distance += time; };
    void inc_num_rh_prefetch(){ m_n_prefetch_reserved_hit++; };
    void inc_num_accessed_once(){ m_n_prefetch_accessed_once++; };
    void inc_late_dist(unsigned time){ m_n_late_distance+=time; };
private:
    bool check_valid(int type, int status) const;
    bool check_fail_valid(int type, int fail) const;


    std::vector< std::vector<unsigned long long> > m_stats;
    // AerialVision cache stats (per-window)
    std::vector< std::vector<unsigned long long> > m_stats_pw;
    std::vector< std::vector<unsigned long long> > m_fail_stats;

    unsigned long long m_cache_port_available_cycles; 
    unsigned long long m_cache_data_port_busy_cycles; 
    unsigned long long m_cache_fill_port_busy_cycles;
    unsigned m_prefetch_stall_cycle;
    unsigned m_n_prefetch_access;
    unsigned m_n_prefetch_hit;
    unsigned m_n_issued_prefetch;
    unsigned m_n_prefetch_distance;
    unsigned m_n_prefetch_reserved_hit;
    unsigned m_n_prefetch_accessed_once;
    unsigned m_n_late_distance;
};

class cache_t {
public:
    virtual ~cache_t() {}
    virtual enum cache_request_status access( new_addr_type addr, mem_fetch *mf, unsigned time, std::list<cache_event> &events ) =  0;

    // accessors for cache bandwidth availability 
    virtual bool data_port_free() const = 0; 
    virtual bool fill_port_free() const = 0; 
};

bool was_write_sent( const std::list<cache_event> &events );
bool was_read_sent( const std::list<cache_event> &events );
bool was_writeallocate_sent( const std::list<cache_event> &events );

/// Baseline cache
/// Implements common functions for read_only_cache and data_cache
/// Each subclass implements its own 'access' function
class baseline_cache : public cache_t {
public:
    baseline_cache( const char *name, cache_config &config, int core_id, int type_id, mem_fetch_interface *memport,
                     enum mem_fetch_status status )
    : m_config(config), m_tag_array(new tag_array(config,core_id,type_id)), 
      m_mshrs(config.m_mshr_entries,config.m_mshr_max_merge),
      m_bandwidth_management(config) 
    {
        init( name, config, memport, status );
    }

    void init( const char *name,
               const cache_config &config,
               mem_fetch_interface *memport,
               enum mem_fetch_status status )
    {
        m_name = name;
        assert(config.m_mshr_type == ASSOC || config.m_mshr_type == SECTOR_ASSOC);
        m_memport=memport;
        m_miss_queue_status = status;
    }

    virtual ~baseline_cache()
    {
        delete m_tag_array;
    }

	void update_cache_parameters(cache_config &config)
	{
		m_config=config;
		m_tag_array->update_cache_parameters(config);
		m_mshrs.check_mshr_parameters(config.m_mshr_entries,config.m_mshr_max_merge);
	}

    virtual enum cache_request_status access( new_addr_type addr, mem_fetch *mf, unsigned time, std::list<cache_event> &events ) =  0;
    /// Sends next request to lower level of memory
    void cycle();
    /// Interface for response from lower memory level (model bandwidth restictions in caller)
    void fill( mem_fetch *mf, unsigned time );
    /// Checks if mf is waiting to be filled by lower memory level
    bool waiting_for_fill( mem_fetch *mf );
    /// Are any (accepted) accesses that had to wait for memory now ready? (does not include accesses that "HIT")
    bool access_ready() const {return m_mshrs.access_ready();}
    /// Pop next ready access (does not include accesses that "HIT")
    mem_fetch *next_access(){return m_mshrs.next_access();}
    // flash invalidate all entries in cache
    void flush(){m_tag_array->flush();}
    void invalidate(){m_tag_array->invalidate();}
    void print(FILE *fp, unsigned &accesses, unsigned &misses) const;
    void display_state( FILE *fp ) const;

    // Stat collection
    const cache_stats &get_stats() const {
        return m_stats;
    }
    unsigned get_stats(enum mem_access_type *access_type, unsigned num_access_type, enum cache_request_status *access_status, unsigned num_access_status)  const{
        return m_stats.get_stats(access_type, num_access_type, access_status, num_access_status);
    }
    void get_sub_stats(struct cache_sub_stats &css) const {
        m_stats.get_sub_stats(css);
    }
    // Clear per-window stats for AerialVision support
    void clear_pw(){
        m_stats.clear_pw();
    }
    // Per-window sub stats for AerialVision support
    void get_sub_stats_pw(struct cache_sub_stats_pw &css) const {
        m_stats.get_sub_stats_pw(css);
    }

    // accessors for cache bandwidth availability 
    bool data_port_free() const { return m_bandwidth_management.data_port_free(); } 
    bool fill_port_free() const { return m_bandwidth_management.fill_port_free(); } 

    // This is a gapping hole we are poking in the system to quickly handle
    // filling the cache on cudamemcopies. We don't care about anything other than
    // L2 state after the memcopy - so just force the tag array to act as though
    // something is read or written without doing anything else.
    void force_tag_access( new_addr_type addr, unsigned time, mem_access_sector_mask_t mask )
    {
        m_tag_array->fill( addr, time, mask, false );
    }

protected:
    // Constructor that can be used by derived classes with custom tag arrays
    baseline_cache( const char *name,
                    cache_config &config,
                    int core_id,
                    int type_id,
                    mem_fetch_interface *memport,
                    enum mem_fetch_status status,
                    tag_array* new_tag_array )
    : m_config(config),
      m_tag_array( new_tag_array ),
      m_mshrs(config.m_mshr_entries,config.m_mshr_max_merge), 
      m_bandwidth_management(config) 
    {
        init( name, config, memport, status );
    }

protected:
    std::string m_name;
    cache_config &m_config;
    tag_array*  m_tag_array;
	mshr_table m_mshrs;
    std::list<mem_fetch*> m_miss_queue;
    enum mem_fetch_status m_miss_queue_status;
    mem_fetch_interface *m_memport;

    struct extra_mf_fields {
        extra_mf_fields()  { m_valid = false;}
        extra_mf_fields( new_addr_type a, new_addr_type ad, unsigned i, unsigned d, const cache_config& m_config)
        {
            m_valid = true;
            m_block_addr = a;
            m_addr = ad;
            m_cache_index = i;
            m_data_size = d;
        	pending_read = m_config.m_mshr_type == SECTOR_ASSOC? m_config.m_line_sz/SECTOR_SIZE : 0;

        }
        bool m_valid;
        new_addr_type m_block_addr;
        new_addr_type m_addr;
        unsigned m_cache_index;
        unsigned m_data_size;
        //this variable is used when a load request generates multiple load transactions
        //For example, a read request from non-sector L1 request sends a request to sector L2
        unsigned pending_read;
    };

    typedef std::map<mem_fetch*,extra_mf_fields> extra_mf_fields_lookup;

    extra_mf_fields_lookup m_extra_mf_fields;

    cache_stats m_stats;

    /// Checks whether this request can be handled on this cycle. num_miss equals max # of misses to be handled on this cycle
    bool miss_queue_full(unsigned num_miss){
    	  return ( (m_miss_queue.size()+num_miss) >= m_config.m_miss_queue_size );
    }
    /// Read miss handler without writeback
    void send_read_request(new_addr_type addr, new_addr_type block_addr, unsigned cache_index, mem_fetch *mf,
    		unsigned time, bool &do_miss, std::list<cache_event> &events, bool read_only, bool wa);
    /// Read miss handler. Check MSHR hit or MSHR available
    void send_read_request(new_addr_type addr, new_addr_type block_addr, unsigned cache_index, mem_fetch *mf,
    		unsigned time, bool &do_miss, bool &wb, evicted_block_info &evicted, std::list<cache_event> &events, bool read_only, bool wa);
    /// Sub-class containing all metadata for port bandwidth management 
    class bandwidth_management 
    {
    public: 
        bandwidth_management(cache_config &config); 

        /// use the data port based on the outcome and events generated by the mem_fetch request 
        void use_data_port(mem_fetch *mf, enum cache_request_status outcome, const std::list<cache_event> &events); 

        /// use the fill port 
        void use_fill_port(mem_fetch *mf); 

        /// called every cache cycle to free up the ports 
        void replenish_port_bandwidth(); 

        /// query for data port availability 
        bool data_port_free() const; 
        /// query for fill port availability 
        bool fill_port_free() const; 
    protected: 
        const cache_config &m_config; 

        int m_data_port_occupied_cycles; //< Number of cycle that the data port remains used 
        int m_fill_port_occupied_cycles; //< Number of cycle that the fill port remains used 
    }; 

    bandwidth_management m_bandwidth_management; 
};

/// Read only cache
class read_only_cache : public baseline_cache {
public:
    read_only_cache( const char *name, cache_config &config, int core_id, int type_id, mem_fetch_interface *memport, enum mem_fetch_status status )
    : baseline_cache(name,config,core_id,type_id,memport,status){}

    /// Access cache for read_only_cache: returns RESERVATION_FAIL if request could not be accepted (for any reason)
    virtual enum cache_request_status access( new_addr_type addr, mem_fetch *mf, unsigned time, std::list<cache_event> &events );

    virtual ~read_only_cache(){}

protected:
    read_only_cache( const char *name, cache_config &config, int core_id, int type_id, mem_fetch_interface *memport, enum mem_fetch_status status, tag_array* new_tag_array )
    : baseline_cache(name,config,core_id,type_id,memport,status, new_tag_array){}
};

/// Data cache - Implements common functions for L1 and L2 data cache
class data_cache : public baseline_cache {
public:
    data_cache( const char *name, cache_config &config,
    			int core_id, int type_id, mem_fetch_interface *memport,
                mem_fetch_allocator *mfcreator, enum mem_fetch_status status,
                mem_access_type wr_alloc_type, mem_access_type wrbk_type )
    			: baseline_cache(name,config,core_id,type_id,memport,status)
    {
        init( mfcreator );
        m_wr_alloc_type = wr_alloc_type;
        m_wrbk_type = wrbk_type;
	m_have_prefetcher=false;
	m_bypass_global_rd_miss=false;
    }

    virtual ~data_cache() {}

    virtual void init( mem_fetch_allocator *mfcreator )
    {
        m_memfetch_creator=mfcreator;

        // Set read hit function
        m_rd_hit = &data_cache::rd_hit_base;

        // Set read miss function
        m_rd_miss = &data_cache::rd_miss_base;

        // Set write hit function
        switch(m_config.m_write_policy){
        // READ_ONLY is now a separate cache class, config is deprecated
        case READ_ONLY:
            assert(0 && "Error: Writable Data_cache set as READ_ONLY\n");
            break; 
        case WRITE_BACK: m_wr_hit = &data_cache::wr_hit_wb; break;
        case WRITE_THROUGH: m_wr_hit = &data_cache::wr_hit_wt; break;
        case WRITE_EVICT: m_wr_hit = &data_cache::wr_hit_we; break;
        case LOCAL_WB_GLOBAL_WT:
            m_wr_hit = &data_cache::wr_hit_global_we_local_wb;
            break;
        default:
            assert(0 && "Error: Must set valid cache write policy\n");
            break; // Need to set a write hit function
        }

        // Set write miss function
        switch(m_config.m_write_alloc_policy){
        case NO_WRITE_ALLOCATE: m_wr_miss = &data_cache::wr_miss_no_wa; break;
		case WRITE_ALLOCATE: m_wr_miss = &data_cache::wr_miss_wa_naive; break;
		case FETCH_ON_WRITE: m_wr_miss = &data_cache::wr_miss_wa_fetch_on_write; break;
		case LAZY_FETCH_ON_READ: m_wr_miss = &data_cache::wr_miss_wa_lazy_fetch_on_read; break;
        default:
            assert(0 && "Error: Must set valid cache write miss policy\n");
            break; // Need to set a write miss function
        }
        bin_num = 12;
        bin_size =256;
        dist_max = 1;
        dist_min = 256;

    }

    virtual enum cache_request_status access( new_addr_type addr,
                                              mem_fetch *mf,
                                              unsigned time,
                                              std::list<cache_event> &events );
    void first_pre_cycle();
    void second_pre_cycle();
    void pre_cycle(Prefetch_Unit* m_prefetcher, unsigned time);
    virtual void fill(mem_fetch* mf, unsigned time);

    std::map<new_addr_type , unsigned > m_pre_issued_map;
    std::map<new_addr_type , unsigned > m_pre_map;
 
    virtual enum cache_request_status pre_access( new_addr_type addr, mem_fetch *mf, unsigned time, std::list<cache_event> &events ) =  0;
	 std::vector<unsigned> get_dist_coll(){
    	return dist_coll;
    }
    unsigned get_max_dist(){return dist_max;}
    unsigned get_min_dist(){return dist_min;}
    unsigned get_dist_size(){return dist_coll.size();}
    void set_prefetch(){
	m_have_prefetcher = true;
	m_mshrs.set_prefetch();
    }
    std::vector<mem_fetch*> breakdown_request(mem_fetch* mf);

protected:
    data_cache( const char *name,
                cache_config &config,
                int core_id,
                int type_id,
                mem_fetch_interface *memport,
                mem_fetch_allocator *mfcreator,
                enum mem_fetch_status status,
                tag_array* new_tag_array,
                mem_access_type wr_alloc_type,
                mem_access_type wrbk_type)
    : baseline_cache(name, config, core_id, type_id, memport,status, new_tag_array)
    {
        init( mfcreator );
        m_wr_alloc_type = wr_alloc_type;
        m_wrbk_type = wrbk_type;
		  fetch_mask = -1;
    }

    mem_access_type m_wr_alloc_type; // Specifies type of write allocate request (e.g., L1 or L2)
    mem_access_type m_wrbk_type; // Specifies type of writeback request (e.g., L1 or L2)

    //! A general function that takes the result of a tag_array probe
    //  and performs the correspding functions based on the cache configuration
    //  The access fucntion calls this function
    enum cache_request_status
        process_tag_probe( bool wr,
                           enum cache_request_status status,
                           new_addr_type addr,
                           unsigned cache_index,
                           mem_fetch* mf,
                           unsigned time,
                           std::list<cache_event>& events );

protected:
    mem_fetch_allocator *m_memfetch_creator;
    bool m_have_prefetcher;
	 bool m_bypass_global_rd_miss;
    // for first_pre_cycle && second_pre_cycle
    std::list<mem_fetch*> m_first_miss_queue;
    std::list<mem_fetch*> m_second_miss_queue;
	 bool (data_cache::*miss_queue_full_fcn)(unsigned);
    bool first_miss_queue_full(unsigned num_miss){
          return ( (m_first_miss_queue.size()+num_miss) >= 2*m_config.m_miss_queue_size );
    }

    bool second_miss_queue_full(unsigned num_miss){
          return ( (m_second_miss_queue.size()+num_miss) >= 2*m_config.m_miss_queue_size );
    }
    void prefetch_send_read_request(new_addr_type addr, new_addr_type block_addr, unsigned cache_index, mem_fetch *mf,
            unsigned time, bool &do_miss, bool &wb, cache_block_t &evicted, std::list<cache_event> &events, bool read_only, bool wa);

    std::vector<unsigned> dist_coll; //only collect num of data. Does not contain data itself.
    unsigned bin_num;	//12
    unsigned bin_size;	//256 or (max-min)/bin_num
    unsigned dist_max;	//256
    unsigned dist_min;	//256
	 unsigned fetch_mask;
    //void make_hist();
    
/// Read miss handler without writeback
    void send_read_request(new_addr_type addr, new_addr_type block_addr, unsigned cache_index, mem_fetch *mf,
    		unsigned time, bool &do_miss, std::list<cache_event> &events, bool read_only, bool wa);
    /// Read miss handler. Check MSHR hit or MSHR available
    void send_read_request(new_addr_type addr, new_addr_type block_addr, unsigned cache_index, mem_fetch *mf,
    		unsigned time, bool &do_miss, bool &wb, evicted_block_info &evicted, std::list<cache_event> &events, bool read_only, bool wa);

	 void send_read_request(mem_fetch* mf, cache_event request, unsigned time, std::list<cache_event> &events);


/////////////////////////////////////////////////////

    // Functions for data cache access
    /// Sends write request to lower level memory (write or writeback)
    void send_write_request( mem_fetch *mf,
                             cache_event request,
                             unsigned time,
                             std::list<cache_event> &events);

    // Member Function pointers - Set by configuration options
    // to the functions below each grouping
    /******* Write-hit configs *******/
    enum cache_request_status
        (data_cache::*m_wr_hit)( new_addr_type addr,
                                 unsigned cache_index,
                                 mem_fetch *mf,
                                 unsigned time,
                                 std::list<cache_event> &events,
                                 enum cache_request_status status );
    /// Marks block as MODIFIED and updates block LRU
    enum cache_request_status
        wr_hit_wb( new_addr_type addr,
                   unsigned cache_index,
                   mem_fetch *mf,
                   unsigned time,
                   std::list<cache_event> &events,
                   enum cache_request_status status ); // write-back
    enum cache_request_status
        wr_hit_wt( new_addr_type addr,
                   unsigned cache_index,
                   mem_fetch *mf,
                   unsigned time,
                   std::list<cache_event> &events,
                   enum cache_request_status status ); // write-through

    /// Marks block as INVALID and sends write request to lower level memory
    enum cache_request_status
        wr_hit_we( new_addr_type addr,
                   unsigned cache_index,
                   mem_fetch *mf,
                   unsigned time,
                   std::list<cache_event> &events,
                   enum cache_request_status status ); // write-evict
    enum cache_request_status
        wr_hit_global_we_local_wb( new_addr_type addr,
                                   unsigned cache_index,
                                   mem_fetch *mf,
                                   unsigned time,
                                   std::list<cache_event> &events,
                                   enum cache_request_status status );
        // global write-evict, local write-back


    /******* Write-miss configs *******/
    enum cache_request_status
        (data_cache::*m_wr_miss)( new_addr_type addr,
                                  unsigned cache_index,
                                  mem_fetch *mf,
                                  unsigned time,
                                  std::list<cache_event> &events,
                                  enum cache_request_status status );
    /// Sends read request, and possible write-back request,
    //  to lower level memory for a write miss with write-allocate
    enum cache_request_status
    	wr_miss_wa_naive( new_addr_type addr,
                        unsigned cache_index,
                        mem_fetch *mf,
                        unsigned time,
                        std::list<cache_event> &events,
                        enum cache_request_status status ); // write-allocate-send-write-and-read-request
	enum cache_request_status
			   wr_miss_wa_fetch_on_write( new_addr_type addr,
							unsigned cache_index,
							mem_fetch *mf,
							unsigned time,
							std::list<cache_event> &events,
							enum cache_request_status status ); // write-allocate with fetch-on-every-write
	enum cache_request_status
				   wr_miss_wa_lazy_fetch_on_read( new_addr_type addr,
								unsigned cache_index,
								mem_fetch *mf,
								unsigned time,
								std::list<cache_event> &events,
								enum cache_request_status status ); // write-allocate with read-fetch-only
	enum cache_request_status
				wr_miss_wa_write_validate( new_addr_type addr,
							unsigned cache_index,
							mem_fetch *mf,
							unsigned time,
							std::list<cache_event> &events,
							enum cache_request_status status ); // write-allocate that writes with no read fetch
    enum cache_request_status
        wr_miss_no_wa( new_addr_type addr,
                       unsigned cache_index,
                       mem_fetch *mf,
                       unsigned time,
                       std::list<cache_event> &events,
                       enum cache_request_status status ); // no write-allocate

    // Currently no separate functions for reads
    /******* Read-hit configs *******/
    enum cache_request_status
        (data_cache::*m_rd_hit)( new_addr_type addr,
                                 unsigned cache_index,
                                 mem_fetch *mf,
                                 unsigned time,
                                 std::list<cache_event> &events,
                                 enum cache_request_status status );
    enum cache_request_status
        rd_hit_base( new_addr_type addr,
                     unsigned cache_index,
                     mem_fetch *mf,
                     unsigned time,
                     std::list<cache_event> &events,
                     enum cache_request_status status );

    /******* Read-miss configs *******/
    enum cache_request_status
        (data_cache::*m_rd_miss)( new_addr_type addr,
                                  unsigned cache_index,
                                  mem_fetch *mf,
                                  unsigned time,
                                  std::list<cache_event> &events,
                                  enum cache_request_status status );
    enum cache_request_status
        rd_miss_base( new_addr_type addr,
                      unsigned cache_index,
                      mem_fetch*mf,
                      unsigned time,
                      std::list<cache_event> &events,
                      enum cache_request_status status );

};

/// This is meant to model the first level data cache in Fermi.
/// It is write-evict (global) or write-back (local) at
/// the granularity of individual blocks
/// (the policy used in fermi according to the CUDA manual)
class l1_cache : public data_cache {
public:
    l1_cache(const char *name, cache_config &config,
            int core_id, int type_id, mem_fetch_interface *memport,
            mem_fetch_allocator *mfcreator, enum mem_fetch_status status )
            : data_cache(name,config,core_id,type_id,memport,mfcreator,status, L1_WR_ALLOC_R, L1_WRBK_ACC)
	 {
		 if(config.have_prefetcher)
			 set_prefetch();
		 fetch_mask = config.fetch_mask;
		 assert(fetch_mask !=0 && "fetch mask should be larger than 0");
	 }

    virtual ~l1_cache(){}

    virtual enum cache_request_status
        access( new_addr_type addr,
                mem_fetch *mf,
                unsigned time,
                std::list<cache_event> &events );
    virtual enum cache_request_status pre_access( new_addr_type addr, mem_fetch *mf, unsigned time, std::list<cache_event> &events );
protected:
    l1_cache( const char *name,
              cache_config &config,
              int core_id,
              int type_id,
              mem_fetch_interface *memport,
              mem_fetch_allocator *mfcreator,
              enum mem_fetch_status status,
              tag_array* new_tag_array )
    : data_cache( name,
                  config,
                  core_id,type_id,memport,mfcreator,status, new_tag_array, L1_WR_ALLOC_R, L1_WRBK_ACC ){}

};

/// Models second level shared cache with global write-back
/// and write-allocate policies
class l2_cache : public data_cache {
public:
    l2_cache(const char *name,  cache_config &config,
            int core_id, int type_id, mem_fetch_interface *memport,
            mem_fetch_allocator *mfcreator, enum mem_fetch_status status )
            : data_cache(name,config,core_id,type_id,memport,mfcreator,status, L2_WR_ALLOC_R, L2_WRBK_ACC){
				m_bypass_global_rd_miss = config.m_bypass_global;}

    virtual ~l2_cache() {}

    virtual enum cache_request_status
        access( new_addr_type addr,
                mem_fetch *mf,
                unsigned time,
                std::list<cache_event> &events );
    virtual enum cache_request_status pre_access( new_addr_type addr, mem_fetch *mf, unsigned time, std::list<cache_event> &events );
 

};
class shader_memory_interface;
class write_buffer{
public:
	write_buffer(const char *name,
			int core_id, int type_id, mem_fetch_interface *memport,
         mem_fetch_allocator *mfcreator, enum mem_fetch_status status, unsigned fm)
			{
				m_name = name;
				m_miss_queue_status = status;
				m_miss_queue_size = 128;
				m_memfetch_creator=mfcreator;
				m_memport = memport;
				m_buffer_tag_array = new buffer_tag_array(core_id, type_id);
				m_wrbk_type=L1WB_WRBK_ACC;
				m_fetch_mask=fm;
	}
	virtual ~write_buffer(){delete m_buffer_tag_array;}
	virtual enum cache_request_status
        access( new_addr_type addr,
                mem_fetch *mf,
                unsigned time,
                std::list<cache_event> &events );
	void send_write_request(mem_fetch* mf, cache_event request, unsigned time, std::list<cache_event>&events, mem_access_sector32_mask_t sector_mask);
	void bypass_read_request(mem_fetch* mf, unsigned time, std::list<cache_event>& events);
	void cycle();

protected:
	buffer_tag_array* m_buffer_tag_array;
	mem_fetch_interface* m_memport;
	std::string m_name;
   std::list<mem_fetch*> m_miss_queue;
	unsigned m_miss_queue_size;
	mem_fetch_allocator* m_memfetch_creator;
	mem_access_type m_wrbk_type;
   enum mem_fetch_status m_miss_queue_status;
	bool miss_queue_full(unsigned int additional){ 
		return m_miss_queue.size()+additional > m_miss_queue_size;
	}
   std::vector<mem_fetch*> breakdown_request(mem_fetch* mf);
	unsigned m_fetch_mask;

};

/*****************************************************************************/

// See the following paper to understand this cache model:
// 
// Igehy, et al., Prefetching in a Texture Cache Architecture, 
// Proceedings of the 1998 Eurographics/SIGGRAPH Workshop on Graphics Hardware
// http://www-graphics.stanford.edu/papers/texture_prefetch/
class tex_cache : public cache_t {
public:
    tex_cache( const char *name, cache_config &config, int core_id, int type_id, mem_fetch_interface *memport,
               enum mem_fetch_status request_status, 
               enum mem_fetch_status rob_status )
    : m_config(config), 
    m_tags(config,core_id,type_id), 
    m_fragment_fifo(config.m_fragment_fifo_entries), 
    m_request_fifo(config.m_request_fifo_entries),
    m_rob(config.m_rob_entries),
    m_result_fifo(config.m_result_fifo_entries)
    {
        m_name = name;
        assert(config.m_mshr_type == TEX_FIFO || config.m_mshr_type == SECTOR_TEX_FIFO );
        assert(config.m_write_policy == READ_ONLY);
        assert(config.m_alloc_policy == ON_MISS);
        m_memport=memport;
        m_cache = new data_block[ config.get_num_lines() ];
        m_request_queue_status = request_status;
        m_rob_status = rob_status;
    }

    /// Access function for tex_cache
    /// return values: RESERVATION_FAIL if request could not be accepted
    /// otherwise returns HIT_RESERVED or MISS; NOTE: *never* returns HIT
    /// since unlike a normal CPU cache, a "HIT" in texture cache does not
    /// mean the data is ready (still need to get through fragment fifo)
    enum cache_request_status access( new_addr_type addr, mem_fetch *mf, unsigned time, std::list<cache_event> &events );
    void cycle();
    /// Place returning cache block into reorder buffer
    void fill( mem_fetch *mf, unsigned time );
    /// Are any (accepted) accesses that had to wait for memory now ready? (does not include accesses that "HIT")
    bool access_ready() const{return !m_result_fifo.empty();}
    /// Pop next ready access (includes both accesses that "HIT" and those that "MISS")
    mem_fetch *next_access(){return m_result_fifo.pop();}
    void display_state( FILE *fp ) const;

    // accessors for cache bandwidth availability - stubs for now 
    bool data_port_free() const { return true; }
    bool fill_port_free() const { return true; }

    // Stat collection
    const cache_stats &get_stats() const {
        return m_stats;
    }
    unsigned get_stats(enum mem_access_type *access_type, unsigned num_access_type, enum cache_request_status *access_status, unsigned num_access_status) const{
        return m_stats.get_stats(access_type, num_access_type, access_status, num_access_status);
    }

    void get_sub_stats(struct cache_sub_stats &css) const{
        m_stats.get_sub_stats(css);
    }
private:
    std::string m_name;
    const cache_config &m_config;

    struct fragment_entry {
        fragment_entry() {}
        fragment_entry( mem_fetch *mf, unsigned idx, bool m, unsigned d )
        {
            m_request=mf;
            m_cache_index=idx;
            m_miss=m;
            m_data_size=d;
        }
        mem_fetch *m_request;     // request information
        unsigned   m_cache_index; // where to look for data
        bool       m_miss;        // true if sent memory request
        unsigned   m_data_size;
    };

    struct rob_entry {
        rob_entry() { m_ready = false; m_time=0; m_request=NULL;}
        rob_entry( unsigned i, mem_fetch *mf, new_addr_type a ) 
        { 
            m_ready=false; 
            m_index=i;
            m_time=0;
            m_request=mf; 
            m_block_addr=a;
        }
        bool m_ready;
        unsigned m_time; // which cycle did this entry become ready?
        unsigned m_index; // where in cache should block be placed?
        mem_fetch *m_request;
        new_addr_type m_block_addr;
    };

    struct data_block {
        data_block() { m_valid = false;}
        bool m_valid;
        new_addr_type m_block_addr;
    };

    // TODO: replace fifo_pipeline with this?
    template<class T> class fifo {
    public:
        fifo( unsigned size ) 
        { 
            m_size=size; 
            m_num=0; 
            m_head=0; 
            m_tail=0; 
            m_data = new T[size];
        }
        bool full() const { return m_num == m_size;}
        bool empty() const { return m_num == 0;}
        unsigned size() const { return m_num;}
        unsigned capacity() const { return m_size;}
        unsigned push( const T &e ) 
        { 
            assert(!full()); 
            m_data[m_head] = e; 
            unsigned result = m_head;
            inc_head(); 
            return result;
        }
        T pop() 
        { 
            assert(!empty()); 
            T result = m_data[m_tail];
            inc_tail();
            return result;
        }
        const T &peek( unsigned index ) const 
        { 
            assert( index < m_size );
            return m_data[index]; 
        }
        T &peek( unsigned index ) 
        { 
            assert( index < m_size );
            return m_data[index]; 
        }
        T &peek() const
        { 
            return m_data[m_tail]; 
        }
        unsigned next_pop_index() const 
        {
            return m_tail;
        }
    private:
        void inc_head() { m_head = (m_head+1)%m_size; m_num++;}
        void inc_tail() { assert(m_num>0); m_tail = (m_tail+1)%m_size; m_num--;}

        unsigned   m_head; // next entry goes here
        unsigned   m_tail; // oldest entry found here
        unsigned   m_num;  // how many in fifo?
        unsigned   m_size; // maximum number of entries in fifo
        T         *m_data;
    };

    tag_array               m_tags;
    fifo<fragment_entry>    m_fragment_fifo;
    fifo<mem_fetch*>        m_request_fifo;
    fifo<rob_entry>         m_rob;
    data_block             *m_cache;
    fifo<mem_fetch*>        m_result_fifo; // next completed texture fetch

    mem_fetch_interface    *m_memport;
    enum mem_fetch_status   m_request_queue_status;
    enum mem_fetch_status   m_rob_status;

    struct extra_mf_fields {
        extra_mf_fields()  { m_valid = false;}
        extra_mf_fields( unsigned i, const cache_config &m_config )
        {
            m_valid = true;
            m_rob_index = i;
            pending_read = m_config.m_mshr_type == SECTOR_TEX_FIFO? m_config.m_line_sz/SECTOR_SIZE : 0;
        }
        bool m_valid;
        unsigned m_rob_index;
        unsigned pending_read;
    };

    cache_stats m_stats;

    typedef std::map<mem_fetch*,extra_mf_fields> extra_mf_fields_lookup;

    extra_mf_fields_lookup m_extra_mf_fields;
};

#endif
