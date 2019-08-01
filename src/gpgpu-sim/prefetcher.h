#ifndef PREFETCHER_H
#define PREFETCHER_H
/*****************************************************************************/
// newly added : Prefetch class
// by jwchoi & jyj
/*****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include "gpu-misc.h"
#include "mem_fetch.h"
#include "../abstract_hardware_model.h"
#include "../tr1_hash_map.h"
#include <iostream>

#include "addrdec.h"

// mem request history buffer
// save mem request from ldst_unit::memory_cycle()
struct prefetcher_mem_hb_t{
    address_type pc;        // address_type = unsigned
    new_addr_type addr;
    unsigned shader_id;
    unsigned cta_id;
    unsigned warp_id;
    unsigned thread_id;
    unsigned inst_uid;       // store uid of instruction that generated current mf
    unsigned timestamp;      // store timestamp of mf
    bool     used_for_prefetch_request;           // check mf is used to generate prefetch request
    long     stride;
};

struct prefetcher_table_t{
    address_type pc;
    unsigned addr;
    unsigned warp_id;
    unsigned thread_id;
    unsigned confidence;
};



#define ADDRALIGN 0xffffff80;
#define MAX_LINE 8
#define ALIGN_32 0xffffffe0;
#define STEP 1

// modified by jwchoi
#define HISTORY_BUFFER_ADDR_INDEX_SIZE  32
#define HISTORY_BUFFER_WARP_INDEX_SIZE  48
#define HISTORY_BUFFER_PC_INDEX_SIZE    32
#define HISTORY_BUFFER_CTA_INDEX_SIZE   8
#define HISTORY_BUFFER_SHADER_INDEX_SIZE   2
#define REQ_Q_MAX_SIZE   8 // cache miss queue size : 8

//added by jyj
//do line prefetch when value is 1
//else use prefetch_stride_pattern_per_warp_matcher()
#define IS_LINE_PREFETCH 1
#define IS_STRIDE_PREFETCH 0
#define IS_SEMI_APRES_PREFETCH 0


class Prefetch_Unit{
public:
    Prefetch_Unit(uint prefetcher_type, uint adaptive_stride, uint knextline, uint cta_x_dim, uint cta_y_dim, uint img_x_size, uint img_y_size, uint img_z_size, uint prefetch_direction){
        init(prefetcher_type, adaptive_stride, knextline, cta_x_dim, cta_y_dim, img_x_size, img_y_size, img_z_size, prefetch_direction);
    }
    ~Prefetch_Unit(){}

// added by jwchoi
// maybe we should add LRU flag for LRU function
// for simple global histroy buffer
typedef std::deque<prefetcher_mem_hb_t> simple_queue_HB;
// for per warp : map's key is warp_id
typedef std::map<unsigned, std::deque<prefetcher_mem_hb_t> > perwarp_HB;
// for per CTA, warp : first key is CTA id, second key is warp id
typedef std::map<unsigned, std::map<unsigned, std::deque<prefetcher_mem_hb_t> > > percta_warp_HB;
// for per CTA, pc : first key is CTA id, second key is pc
typedef std::map<unsigned, std::map<unsigned, std::deque<prefetcher_mem_hb_t> > > percta_pc_HB;
// for per shader, warp : first key is shader id, second key is warp id
typedef std::map<unsigned, std::map<unsigned, std::deque<prefetcher_mem_hb_t> > > pershader_warp_HB;

// jwchoi added (0711)
// pattern History Buffer
// <pc, <pattern number, pattern value>>
// pattern number is not specified yet (just inorder)
typedef std::map<unsigned, std::map<unsigned, std::vector<long long> > > perpc_pattern_HB;

// jwchoi added (0615)
// for per shader, warp : first key is shader id, second key is warp id
// not added yet : since prefetcher is in one shader core, shader id may not be needed
// --> shader id fixed ? --> then we can just use percta_warp_HB
typedef std::map<unsigned, std::map<unsigned, std::map<unsigned, std::deque<prefetcher_mem_hb_t> > > > pershader_cta_warp_HB;

    void init(uint prefetcher_type, 
              uint adaptive_stride, 
              uint knextline, 
              uint cta_x_dim, 
              uint cta_y_dim, 
              uint img_x_size, 
              uint img_y_size, 
              uint img_z_size, 
              uint prefetch_direction)
    {
        //jwchoi added for config (1122)
        m_prefetcher_type = prefetcher_type;    // 0:Nextline, 1:Stride
        ADAPTIVE_ROW_CHANGE = adaptive_stride;
        m_knextline = knextline;

        m_cta_x_dim = cta_x_dim;
        m_cta_y_dim = cta_y_dim;

        m_img_x_size = img_x_size;  // img size - we will use it later
        m_img_y_size = img_y_size;  // img size - we will use it later
        m_img_z_size = img_z_size;  // img size - we will use it later

        m_prefetch_direction = prefetch_direction;

        m_start_address = 2147483648;

        row_change_count = 0;
        row_change_threshold = (img_x_size/cta_x_dim)-1;

        // m_start_address = minimum address (for first kernel) 
        // for debugging
        //std::cout<<"Prefetcher type : "<<m_prefetcher_type<<std::endl;
        //std::cout<<"Prefetcher nextline : "<<m_knextline<<std::endl;

        m_req_q.clear();

        // added by jwchoi
        // max size of mem_hb is HISTORY_BUFFER_ADDR_INDEX_SIZE
        // for global queue : just save all request
        // it will not be used in real hardware
        // directly save warp to corresponding warp buffer
        prefetch_mem_hb.clear();
        prefetch_mem_hb_perwarp.clear();
        prefetch_mem_hb_percta_perwarp.clear();
        // jwchoi added for APRES (0709)
        prefetch_mem_hb_percta_perpc.clear();
        queue_prefetch_mem_hb_percta_perpc.clear();
        prefetch_mem_hb_pershader_perwarp.clear();
    }

    //void reinit()
    //{
    //    init();
    //}

    // modified by jwchoi
    // check if number of saved request exceeds capacity
    bool history_buffer_is_full_request(){return prefetch_mem_hb.size()>=HISTORY_BUFFER_ADDR_INDEX_SIZE;}

    // check if number of saved warp exceeds capacity
    bool history_buffer_is_full_perwarp_request(unsigned wid)
    {
        return prefetch_mem_hb_perwarp[wid].size()>=HISTORY_BUFFER_ADDR_INDEX_SIZE;
    }
    /*bool history_buffer_is_full_perwarp()
    {
        return prefetch_mem_hb_perwarp.size()>=HISTORY_BUFFER_WARP_INDEX_SIZE;
    }*/

    // check if number of saved cta and warp exceeds capacity
    bool history_buffer_is_full_percta_perwarp_addr_request(unsigned cid, unsigned wid)
    {
        return prefetch_mem_hb_percta_perwarp[cid][wid].size()>=HISTORY_BUFFER_ADDR_INDEX_SIZE;
    }
    bool history_buffer_is_full_percta_perpc_addr_request(unsigned cid, unsigned wid)
    {
        return prefetch_mem_hb_percta_perpc[cid][wid].size()>=HISTORY_BUFFER_ADDR_INDEX_SIZE;
    }
    bool history_buffer_is_full_queue_percta_perpc_addr_request(unsigned cid, unsigned wid)
    {
        return queue_prefetch_mem_hb_percta_perpc[cid][wid].size()>=HISTORY_BUFFER_ADDR_INDEX_SIZE;
    }
    bool history_buffer_is_full_percta_perwarp_warp_request(unsigned cid)
    {
        return prefetch_mem_hb_percta_perwarp[cid].size()>=HISTORY_BUFFER_WARP_INDEX_SIZE;
    }
    bool history_buffer_is_full_percta_perpc_pc_request(unsigned cid)
    {
        return prefetch_mem_hb_percta_perpc[cid].size()>=HISTORY_BUFFER_PC_INDEX_SIZE;
    }
    bool history_buffer_is_full_percta_perwarp_cta_request()
    {
        return prefetch_mem_hb_percta_perwarp.size()>=HISTORY_BUFFER_CTA_INDEX_SIZE;
    }

    bool history_buffer_is_full_pershader_perwarp_addr_request(unsigned sid, unsigned wid)
    {
        return prefetch_mem_hb_pershader_perwarp[sid][wid].size()>=HISTORY_BUFFER_ADDR_INDEX_SIZE;
    }
    bool history_buffer_is_full_pershader_perwarp_warp_request(unsigned sid)
    {
        return prefetch_mem_hb_pershader_perwarp[sid].size()>=HISTORY_BUFFER_WARP_INDEX_SIZE;
    }
    bool history_buffer_is_full_pershader_perwarp_shader_request()
    {
        return prefetch_mem_hb_pershader_perwarp.size()>=HISTORY_BUFFER_SHADER_INDEX_SIZE;
    }

    // check if next address is next row block
    bool check_row_change(new_addr_type demand_addr)
    {
        unsigned x_coordinate = ((demand_addr - m_start_address)%(m_img_x_size*4))/4; // 4 = size of int
        // if x_coordinate >= img_size - cta_size --> prefetch next row block
        // 32 is cache line size/4 or cta size
        if(x_coordinate >= (m_img_x_size-m_cta_x_dim*m_knextline)){
            return true;
        }
        else{
            return false;
        }
    }

    // check if next address is next row block
    bool check_col_change(new_addr_type demand_addr)
    {
        unsigned y_coordinate = ((demand_addr - m_start_address)/(m_img_x_size*4))/4; // 4 = size of int
        // if x_coordinate >= img_size - cta_size --> prefetch next row block
        // 32 is cache line size/4 or cta size
        if(y_coordinate >= (m_img_y_size-m_cta_y_dim*m_knextline)){
            return true;
        }
        else{
            return false;
        }
    }

    // modified by jwchoi
    // save memory request to cache(=mem load request from core) to history buffer
    // todo
    // 1. cache address = memory address ?
    // 2. thread stride is always 1 ? --> if it is true, then we don't have to save thread id
    // 2-1. maybe we can save just 2 or 3 thread ids to check thread's stride (in one warp)
    // 3. should we save CTA id and shader core id ? --> yes (180601) but we don't know how to
    // calculate stride when saving address ?
    // save confidence ?

    void update_address_history_for_prefetch(mem_fetch *mf)
    {
        struct prefetcher_mem_hb_t temp_mem_hb;
        temp_mem_hb.pc = mf->get_pc();
        temp_mem_hb.addr = mf->get_addr();
        temp_mem_hb.shader_id = mf->get_sid();
        //temp_mem_hb.cta_id = inst.get_cta_id();
        temp_mem_hb.cta_id = (unsigned)-1;  // false value

        temp_mem_hb.warp_id = mf->get_wid();
        temp_mem_hb.stride = temp_mem_hb.addr - (prefetch_mem_hb.back()).addr;
        // if buffer full : pop front (pop unused one) (LRU)
        if( history_buffer_is_full_request()){
             prefetch_mem_hb.pop_front();
        }
        prefetch_mem_hb.push_back(temp_mem_hb);
    }

    // should be modified
    void update_address_history_for_prefetch_perwarp(mem_fetch *mf)
    {
        struct prefetcher_mem_hb_t temp_mem_hb;
        unsigned wid = 0;

        temp_mem_hb.pc = mf->get_pc();
        temp_mem_hb.addr = mf->get_addr();
        temp_mem_hb.shader_id = mf->get_sid();
        //temp_mem_hb.cta_id = inst.get_cta_id();
        temp_mem_hb.cta_id = (unsigned)-1;

        wid = mf->get_wid();
        temp_mem_hb.warp_id = wid;

        // if warp id is first executed, insert warp id as a key and set stride as invalid value (= -1)
        // check if total warp index exceeds buffer warp index size
        if(!(prefetch_mem_hb_perwarp.count(wid))){
            temp_mem_hb.stride = 128;
            // check if total warp index exceeds buffer warp index size
            // don't check it now : maybe it's not a problem
            // maybe we can provide sufficient area(=space) for warp and cta
            //if(history_buffer_is_full_perwarp()){
            //    pop
            //}

            //simple_queue_HB new_warp_history;
            //new_warp_history.push_back(temp_mem_hb);
            //prefetch_mem_hb_perwarp.insert(std::pair<int, simple_queue_HB>(wid, new_warp_history));
            // it can be coded by simply doing like this
            prefetch_mem_hb_perwarp[wid].push_back(temp_mem_hb);
        }
        else{
            temp_mem_hb.stride = temp_mem_hb.addr - (prefetch_mem_hb_perwarp[wid].back()).addr;
            // if number of warps in buffer exceeds  : pop front (pop unused warp)
            // don't check warp & cta --> just check whether request buffer exceeds it limit
            if( history_buffer_is_full_perwarp_request(wid) ){
                prefetch_mem_hb_perwarp[wid].pop_front();
            }
            prefetch_mem_hb_perwarp[wid].push_back(temp_mem_hb);
        }
    }

    void update_address_history_for_prefetch_percta_perwarp(mem_fetch *mf)
    {
        struct prefetcher_mem_hb_t temp_mem_hb;
        unsigned wid;
        unsigned cid;

        temp_mem_hb.pc = mf->get_pc();
        temp_mem_hb.addr = mf->get_addr();
        temp_mem_hb.shader_id = mf->get_sid();

        wid = mf->get_wid();
        cid = mf->get_mf_cta_id();
        temp_mem_hb.warp_id = wid;
        temp_mem_hb.cta_id = cid;

        // todo
        // if cta id is first executed, insert cta id as a key and set stride as invalid value
        if(!(prefetch_mem_hb_percta_perwarp.count(cid))){
            temp_mem_hb.stride = 128;

            //simple_queue_HB new_warp_history;
            //new_warp_history.push_back(temp_mem_hb);
            // create new CTA
            //perwarp_HB new_cta_history;
            //new_cta_history.insert(std::pair<int, simple_queue_HB>(wid, new_warp_history))
            // insert CTA into CTA map
            //prefetch_mem_hb_perwarp.insert(std::pair<int, perwarp_HB>(cid, new_cta_history));
            // if CTA index exceeds : remove oldest used CTA (LRU)
            // don't check it now : maybe it's not a problem
            // maybe we can provide sufficient area(=space) for warp and cta
            //if(history_buffer_is_full_percta()){
            //    pop
            //}
            prefetch_mem_hb_percta_perwarp[cid][wid].push_back(temp_mem_hb);
        }
        // else if warp id is first executed, insert warp id as a key and set stride as invalid value
        else{
            // check if it is first warp in CTA
            if(!(prefetch_mem_hb_percta_perwarp[cid].count(wid))){
                temp_mem_hb.stride = 128;

                // don't check it now : maybe it's not a problem
                // maybe we can provide sufficient area(=space) for warp and cta
                //if(history_buffer_is_full_percta_perwarp(cid)){
                //    pop
                //}
                //simple_queue_HB new_warp_history;
                //new_warp_history.push_back(temp_mem_hb);
                //prefetch_mem_hb_perwarp.insert(std::pair<int, simple_queue_HB>(wid, new_warp_history));
                prefetch_mem_hb_percta_perwarp[cid][wid].push_back(temp_mem_hb);
            }
            else{
                temp_mem_hb.stride = temp_mem_hb.addr - (prefetch_mem_hb_percta_perwarp[temp_mem_hb.cta_id][temp_mem_hb.warp_id].back()).addr;
                // if number of CTAs in buffer exceeds  : pop front (pop unused CTA)
                if( history_buffer_is_full_percta_perwarp_addr_request(cid, wid)){
                    prefetch_mem_hb_percta_perwarp[cid][wid].pop_front();
                }
                prefetch_mem_hb_percta_perwarp[cid][wid].push_back(temp_mem_hb);
            }
        }
        // if number of warps in CTA buffer exceeds  : pop front (pop unused warp)
        // it won't occur
    }

    void update_address_history_for_prefetch_pershader_perwarp(mem_fetch *mf)
    {
        struct prefetcher_mem_hb_t temp_mem_hb;
        unsigned wid;
        unsigned sid;

        temp_mem_hb.pc = mf->get_pc();
        temp_mem_hb.addr = mf->get_addr();
        sid = mf->get_sid();
        temp_mem_hb.shader_id = sid;
        //temp_mem_hb.cta_id = inst.get_cta_id();
        temp_mem_hb.cta_id = 100;

        wid = mf->get_wid();
        temp_mem_hb.warp_id = wid;

        // todo
        // if cta id is first executed, insert cta id as a key and set stride as invalid value
        if(!(prefetch_mem_hb_percta_perwarp.count(sid))){
            temp_mem_hb.stride = 128;

            //simple_queue_HB new_warp_history;
            //new_warp_history.push_back(temp_mem_hb);
            // create new CTA
            //perwarp_HB new_cta_history;
            //new_cta_history.insert(std::pair<int, simple_queue_HB>(wid, new_warp_history))
            // insert CTA into CTA map
            //prefetch_mem_hb_perwarp.insert(std::pair<int, perwarp_HB>(sid, new_cta_history));
            // if CTA index exceeds : remove oldest used CTA (LRU)
            // don't check it now : maybe it's not a problem
            // maybe we can provide sufficient area(=space) for warp and cta
            //if(history_buffer_is_full_percta()){
            //    pop
            //}
            prefetch_mem_hb_pershader_perwarp[sid][wid].push_back(temp_mem_hb);
        }
        // else if warp id is first executed, insert warp id as a key and set stride as invalid value
        else{
            // check if it is first warp in CTA
            if(!(prefetch_mem_hb_pershader_perwarp[sid].count(wid))){
                temp_mem_hb.stride = 128;

                // don't check it now : maybe it's not a problem
                // maybe we can provide sufficient area(=space) for warp and cta
                //if(history_buffer_is_full_percta_perwarp(sid)){
                //    pop
                //}
                //simple_queue_HB new_warp_history;
                //new_warp_history.push_back(temp_mem_hb);
                //prefetch_mem_hb_perwarp.insert(std::pair<int, simple_queue_HB>(wid, new_warp_history));
                prefetch_mem_hb_pershader_perwarp[sid][wid].push_back(temp_mem_hb);
            }
            else{
                temp_mem_hb.stride = temp_mem_hb.addr - (prefetch_mem_hb_pershader_perwarp[sid][wid].back()).addr;
                // if number of CTAs in buffer exceeds  : pop front (pop unused CTA)
                if( history_buffer_is_full_pershader_perwarp_addr_request(sid, wid)){
                    prefetch_mem_hb_pershader_perwarp[sid][wid].pop_front();
                }
                prefetch_mem_hb_pershader_perwarp[sid][wid].push_back(temp_mem_hb);
            }
        }
        // if number of warps in CTA buffer exceeds  : pop front (pop unused warp)
        // it won't occur
    }


    // APRES
    void update_address_history_for_prefetch_percta_perpc(mem_fetch *mf)
    {
        struct prefetcher_mem_hb_t temp_mem_hb;
        unsigned wid;
        unsigned cid;
        address_type pc_id;

        pc_id = mf->get_pc();
        temp_mem_hb.pc = pc_id;
        temp_mem_hb.addr = mf->get_addr();
        temp_mem_hb.shader_id = mf->get_sid();

        wid = mf->get_wid();
        cid = mf->get_mf_cta_id();
        temp_mem_hb.warp_id = wid;
        temp_mem_hb.cta_id = cid;

        // todo
        // if cta id is first executed, insert cta id as a key and set stride as invalid value
        if(!(prefetch_mem_hb_percta_perpc.count(cid))){
            temp_mem_hb.stride = 128;
            prefetch_mem_hb_percta_perpc[cid][pc_id].push_back(temp_mem_hb);
        }
        // else if warp id is first executed, insert warp id as a key and set stride as invalid value
        else{
            // check if it is first pc in CTA
            if(!(prefetch_mem_hb_percta_perpc[cid].count(pc_id))){
                temp_mem_hb.stride = 128;

                // don't check it now : maybe it's not a problem
                // maybe we can provide sufficient area(=space) for warp and cta
                //if(history_buffer_is_full_percta_perwarp(cid)){
                //    pop
                //}
                //simple_queue_HB new_warp_history;
                //new_warp_history.push_back(temp_mem_hb);
                //prefetch_mem_hb_perwarp.insert(std::pair<int, simple_queue_HB>(wid, new_warp_history));
                prefetch_mem_hb_percta_perpc[cid][pc_id].push_back(temp_mem_hb);
            }
            else{
                // APRES
                // Todo : calculate stride with leading warp & add confidence
                if (wid != (prefetch_mem_hb_percta_perpc[temp_mem_hb.cta_id][temp_mem_hb.pc].back()).warp_id){
                    temp_mem_hb.stride = (temp_mem_hb.addr - (prefetch_mem_hb_percta_perpc[temp_mem_hb.cta_id][temp_mem_hb.pc].back()).addr)/(wid - (prefetch_mem_hb_percta_perpc[temp_mem_hb.cta_id][temp_mem_hb.pc].back()).warp_id);
                }
                else{
                    temp_mem_hb.stride = temp_mem_hb.addr - (prefetch_mem_hb_percta_perpc[temp_mem_hb.cta_id][temp_mem_hb.pc].back()).addr;
                }
                // if number of CTAs in buffer exceeds  : pop front (pop unused CTA)
                // share same function
                if( history_buffer_is_full_percta_perpc_addr_request(cid, pc_id)){
                    prefetch_mem_hb_percta_perpc[cid][pc_id].pop_front();
                }
                prefetch_mem_hb_percta_perpc[cid][pc_id].push_back(temp_mem_hb);
            }
        }
        // if number of warps in CTA buffer exceeds  : pop front (pop unused warp)
        // it won't occur
    }


    // save instruction's uid to check whether cache is stalled or not
    // if consecutive inst is stored, same inst_uid will be inserted
    // if same inst_uid -> don't store
    // main save function
    void update_address_history_for_prefetch_percta_perpc_store_uid(mem_fetch *mf, unsigned inst_uid)
    {
        prefetcher_mem_hb_t temp_mem_hb;
        unsigned wid;
        unsigned cid;
        address_type pc_id;

        pc_id = mf->get_pc();
        temp_mem_hb.pc = pc_id;
        temp_mem_hb.addr = mf->get_addr();
        temp_mem_hb.shader_id = mf->get_sid();

        wid = mf->get_wid();
        cid = mf->get_mf_cta_id();
        temp_mem_hb.warp_id = wid;
        temp_mem_hb.cta_id = cid;

        temp_mem_hb.inst_uid = inst_uid;
        temp_mem_hb.timestamp = mf->get_timestamp();

        // added by jwchoi
        temp_mem_hb.used_for_prefetch_request = false;

        // todo
        // if cta id is first executed, insert cta id as a key and set stride as invalid value
        if(!(prefetch_mem_hb_percta_perpc.count(cid))){
            temp_mem_hb.stride = 128;
            prefetch_mem_hb_percta_perpc[cid][pc_id].push_back(temp_mem_hb);
        }
        // else if warp id is first executed, insert warp id as a key and set stride as invalid value
        else{
            // check if it is first pc in CTA
            if(!(prefetch_mem_hb_percta_perpc[cid].count(pc_id))){
                temp_mem_hb.stride = 128;
                prefetch_mem_hb_percta_perpc[cid][pc_id].push_back(temp_mem_hb);
            }
            else{
                // APRES
                // Todo : calculate stride with leading warp & add confidence
                if (wid != (prefetch_mem_hb_percta_perpc[cid][pc_id].back()).warp_id){
                    if(temp_mem_hb.addr > (prefetch_mem_hb_percta_perpc[cid][pc_id].back()).addr){
                        long temp_stride = temp_mem_hb.addr - (prefetch_mem_hb_percta_perpc[cid][pc_id].back()).addr;
                        temp_mem_hb.stride = temp_stride/((int)wid - (int)(prefetch_mem_hb_percta_perpc[cid][pc_id].back()).warp_id);
                    }
                    else{
                        long temp_stride = (prefetch_mem_hb_percta_perpc[cid][pc_id].back()).addr - temp_mem_hb.addr;
                        temp_mem_hb.stride = (temp_stride/((int)wid - (int)(prefetch_mem_hb_percta_perpc[cid][pc_id].back()).warp_id))*(-1);
                    }
                }
                else{
                    temp_mem_hb.stride = temp_mem_hb.addr - (prefetch_mem_hb_percta_perpc[cid][pc_id].back()).addr;
                }
                // if number of CTAs in buffer exceeds  : pop front (pop unused CTA)
                // share same function
                if( history_buffer_is_full_percta_perpc_addr_request(cid, pc_id)){
                    prefetch_mem_hb_percta_perpc[cid][pc_id].pop_front();
                }

                // if instruction's uid and address is same, decide "cache is stalled" --> don't store request's information
                if((inst_uid != (prefetch_mem_hb_percta_perpc[cid][pc_id].back()).inst_uid) || (temp_mem_hb.addr != (prefetch_mem_hb_percta_perpc[cid][pc_id].back()).addr)){
                    prefetch_mem_hb_percta_perpc[cid][pc_id].push_back(temp_mem_hb);
                }
                //prefetch_mem_hb_percta_perpc[cid][pc_id].push_back(temp_mem_hb);

                //if((inst_uid != (prefetch_mem_hb_percta_perpc[cid][pc_id].back()).inst_uid)){
                //    prefetch_mem_hb_percta_perpc[cid][pc_id].push_back(temp_mem_hb);
                //}
                //prefetch_mem_hb_percta_perpc[cid][pc_id].push_back(temp_mem_hb);
            }
        }
        // if number of warps in CTA buffer exceeds  : pop front (pop unused warp)
        // it won't occur
    }

    // for inst empty case, jwchoi added (0802)
    // save same history, but if mf is used for prefetch request generation, pop that mf
    // it will be done in do prefetch
    void update_address_history_for_prefetch_queue_percta_perpc_store_uid(mem_fetch *mf, unsigned inst_uid)
    {
        prefetcher_mem_hb_t temp_mem_hb;
        unsigned wid;
        unsigned cid;
        address_type pc_id;

        pc_id = mf->get_pc();
        temp_mem_hb.pc = pc_id;
        temp_mem_hb.addr = mf->get_addr();
        temp_mem_hb.shader_id = mf->get_sid();

        wid = mf->get_wid();
        cid = mf->get_mf_cta_id();
        temp_mem_hb.warp_id = wid;
        temp_mem_hb.cta_id = cid;

        temp_mem_hb.inst_uid = inst_uid;
        temp_mem_hb.timestamp = mf->get_timestamp();

        // todo
        // if cta id is first executed, insert cta id as a key and set stride as invalid value
        if(!(queue_prefetch_mem_hb_percta_perpc.count(cid))){
            temp_mem_hb.stride = 128;
            queue_prefetch_mem_hb_percta_perpc[cid][pc_id].push_back(temp_mem_hb);
        }
        // else if warp id is first executed, insert warp id as a key and set stride as invalid value
        else{
            // check if it is first pc in CTA
            if(!(queue_prefetch_mem_hb_percta_perpc[cid].count(pc_id))){
                temp_mem_hb.stride = 128;
                queue_prefetch_mem_hb_percta_perpc[cid][pc_id].push_back(temp_mem_hb);
            }
            else{
                // APRES
                // Todo : calculate stride with leading warp & add confidence
                if (wid != (queue_prefetch_mem_hb_percta_perpc[cid][pc_id].back()).warp_id){
                    temp_mem_hb.stride = (temp_mem_hb.addr - (queue_prefetch_mem_hb_percta_perpc[cid][pc_id].back()).addr)/(wid - (queue_prefetch_mem_hb_percta_perpc[cid][pc_id].back()).warp_id);
                }
                else{
                    temp_mem_hb.stride = temp_mem_hb.addr - (queue_prefetch_mem_hb_percta_perpc[cid][pc_id].back()).addr;
                }
                // if number of CTAs in buffer exceeds  : pop front (pop unused CTA)
                // share same function
                if( history_buffer_is_full_queue_percta_perpc_addr_request(cid, pc_id)){
                    queue_prefetch_mem_hb_percta_perpc[cid][pc_id].pop_front();
                }

                // if instruction's uid and address is same, decide "cache is stalled" --> don't store request's information
                if((inst_uid != (queue_prefetch_mem_hb_percta_perpc[cid][pc_id].back()).inst_uid) || (temp_mem_hb.addr != (queue_prefetch_mem_hb_percta_perpc[cid][pc_id].back()).addr)){
                    queue_prefetch_mem_hb_percta_perpc[cid][pc_id].push_back(temp_mem_hb);
                }
            }
        }
    }


    // jwchoi : adding pattern matcher => only work for 1 or 4 warp?
    void update_address_history_for_prefetch_percta_perpc_forpattern(mem_fetch *mf)
    {
        struct prefetcher_mem_hb_t temp_mem_hb;
        unsigned wid = 0;
        unsigned cid = 0;
        address_type pc_id = 0; // pc_id = address_type = unsigned

        pc_id = mf->get_pc();
        temp_mem_hb.pc = pc_id;
        temp_mem_hb.addr = mf->get_addr();
        temp_mem_hb.shader_id = mf->get_sid();

        wid = mf->get_wid();
        cid = mf->get_mf_cta_id();
        temp_mem_hb.warp_id = wid;
        temp_mem_hb.cta_id = cid;

        // todo
        // if cta id is first executed, insert cta id as a key and set stride as invalid value
        if(!(prefetch_mem_hb_percta_perpc.count(cid))){
            temp_mem_hb.stride = 128;
            prefetch_mem_hb_percta_perpc[cid][pc_id].push_back(temp_mem_hb);
        }
        // else if warp id is first executed, insert warp id as a key and set stride as invalid value
        else{
            // check if it is first pc in CTA
            if(!(prefetch_mem_hb_percta_perpc[cid].count(pc_id))){
                temp_mem_hb.stride = 128;

                // don't check it now : maybe it's not a problem
                // maybe we can provide sufficient area(=space) for warp and cta
                //if(history_buffer_is_full_percta_perwarp(cid)){
                //    pop
                //}
                //simple_queue_HB new_warp_history;
                //new_warp_history.push_back(temp_mem_hb);
                //prefetch_mem_hb_perwarp.insert(std::pair<int, simple_queue_HB>(wid, new_warp_history));
                prefetch_mem_hb_percta_perpc[cid][pc_id].push_back(temp_mem_hb);
            }
            else{
                // APRES
                // Todo : calculate stride with leading warp & add confidence
                if (wid != (prefetch_mem_hb_percta_perpc[temp_mem_hb.cta_id][temp_mem_hb.pc].back()).warp_id){
                    temp_mem_hb.stride = (temp_mem_hb.addr - (prefetch_mem_hb_percta_perpc[temp_mem_hb.cta_id][temp_mem_hb.pc].back()).addr)/(wid - (prefetch_mem_hb_percta_perpc[temp_mem_hb.cta_id][temp_mem_hb.pc].back()).warp_id);
                }
                else{
                    temp_mem_hb.stride = temp_mem_hb.addr - (prefetch_mem_hb_percta_perpc[temp_mem_hb.cta_id][temp_mem_hb.pc].back()).addr;
                }
                // if number of CTAs in buffer exceeds  : pop front (pop unused CTA)
                // share same function
                if( history_buffer_is_full_percta_perpc_addr_request(cid, pc_id)){
                    prefetch_mem_hb_percta_perpc[cid][pc_id].pop_front();
                }
                prefetch_mem_hb_percta_perpc[cid][pc_id].push_back(temp_mem_hb);
            }
        }
        // if number of warps in CTA buffer exceeds  : pop front (pop unused warp)
        // it won't occur
    }

    //added by jyj
    //return next line address
    new_addr_type prefetch_knextline(mem_access_t access);
    // jwchoi added 0810
    // this will be used to check earliest not-used mf
    // index for checking request
    prefetcher_mem_hb_t get_oldest_history_mf_check_all(int &index){
        //std::map<unsigned, std::map<unsigned, std::deque<prefetcher_mem_hb_t> > >::iterator it_history;

        prefetcher_mem_hb_t temp_oldest_mem;
        temp_oldest_mem.cta_id = (unsigned)-1;  // set false value first

        unsigned oldest_time    = (unsigned)-1;
        unsigned oldest_cta     = (unsigned)-1;
        int oldest_content      = 0;
        address_type oldest_pc  = (unsigned)-1; // address_type = unsigned

        if(prefetch_mem_hb_percta_perpc.size() > 0){
            std::map<unsigned, std::map<unsigned, std::deque<prefetcher_mem_hb_t> > >::iterator it_history;
            for(it_history = prefetch_mem_hb_percta_perpc.begin(); it_history != prefetch_mem_hb_percta_perpc.end(); ++it_history){
                //std::cout<<"cta id : "<<it_history->first<<std::endl;
                std::map<unsigned, std::deque<prefetcher_mem_hb_t> >::iterator it_perpc;
                if(prefetch_mem_hb_percta_perpc[it_history->first].size() > 0){
                    for(it_perpc = (it_history->second).begin(); it_perpc != (it_history->second).end(); ++it_perpc){
                        //std::cout<<"pc : "<<it_perpc->first<<std::endl;
                        if(prefetch_mem_hb_percta_perpc[it_history->first][it_perpc->first].size() > 0){
                            for(unsigned i = 0; i<prefetch_mem_hb_percta_perpc[it_history->first][it_perpc->first].size(); ++i){
                                if(prefetch_mem_hb_percta_perpc[it_history->first][it_perpc->first][i].used_for_prefetch_request == false){
                                    if(prefetch_mem_hb_percta_perpc[it_history->first][it_perpc->first][i].timestamp < oldest_time){
                                        oldest_time = prefetch_mem_hb_percta_perpc[it_history->first][it_perpc->first][i].timestamp;
                                        oldest_cta = it_history->first;
                                        oldest_pc = it_perpc->first;
                                        oldest_content = i;
                                    }
                                    break;
                                }
                            }
                        }
                    }
                }
            }
            if(oldest_cta != (unsigned)-1){
                temp_oldest_mem = prefetch_mem_hb_percta_perpc[oldest_cta][oldest_pc][oldest_content];
                index = oldest_content;
            }
        }
        return temp_oldest_mem;
    }

    // jwchoi added 0810
    // this will be used to check earliest not-used mf
    // index for checking request
    prefetcher_mem_hb_t get_oldest_history_mf_check_per_cta_pc(unsigned cid, unsigned pc, int &index){
        //std::map<unsigned, std::map<unsigned, std::deque<prefetcher_mem_hb_t> > >::iterator it_history;

        prefetcher_mem_hb_t temp_oldest_mem;
        temp_oldest_mem.cta_id = (unsigned)-1;  // set false value first

        //unsigned oldest_time    = (unsigned)-1;
        unsigned oldest_cta     = (unsigned)-1;
        int oldest_content      = 0;
        address_type oldest_pc  = (unsigned)-1; // address_type = unsigned

        if(prefetch_mem_hb_percta_perpc.size() > 0){
            if(prefetch_mem_hb_percta_perpc[cid][pc].size() > 0){
                for(unsigned i = 0; i<prefetch_mem_hb_percta_perpc[cid][pc].size(); ++i){
                    if(prefetch_mem_hb_percta_perpc[cid][pc][i].used_for_prefetch_request == false){
                        oldest_cta = cid;
                        oldest_pc = pc;
                        oldest_content = i;

                        break;
                    }
                }
            }

            if(oldest_cta != (unsigned)-1){
                temp_oldest_mem = prefetch_mem_hb_percta_perpc[oldest_cta][oldest_pc][oldest_content];
                index = oldest_content;
            }
        }
        return temp_oldest_mem;
    }


    void do_prefetch(const warp_inst_t &inst, mem_access_t access, unsigned data_size);

	 //added by jyj
    //generate mem access with predicted address
    //put generated mem access into prefetch queue
    // modified by jwchoi (0608)
    // return true if request is used for generation prefetch request
    bool gen_prefetch_request(new_addr_type addr, unsigned cta_id, unsigned warp_id, unsigned data_size);
	
    bool is_req_in_queue(new_addr_type addr)
    {
        //std::list<mem_access_t*>::iterator it = m_req_q.begin();
        //std::list<mem_access_t>::iterator it;
        std::list<std::pair<mem_access_t, std::pair<unsigned, unsigned>>>::iterator it;
        for(it = m_req_q.begin();it!=m_req_q.end();it++){
            if(((*it).first).get_addr()==addr)
                return true;
        }
        return false;
    }


    // modified by jwchoi
    const mem_access_t &cal_pref_q_top_access() {return (m_req_q.front()).first;}
    unsigned cal_pref_q_top_cta_id() {return ((m_req_q.front()).second).first;}
    unsigned cal_pref_q_top_warp_id() {return ((m_req_q.front()).second).second;}
    void del_req_from_top() {m_req_q.pop_front();}
    bool queue_empty() {return !m_req_q.size();}
    bool queue_full() {
        if(m_req_q.size() >= REQ_Q_MAX_SIZE){
            return true;
        }
        else {
            return false;
        }
    }
    int req_q_size() {return m_req_q.size();}
    void print_req_q_size() {
        std::cout<<"request queue size : "<<m_req_q.size()<<std::endl;
    }

    mem_fetch* front_pre2icnt_q() { return m_pre2icnt_q.front(); }
    bool pre2icnt_q_empty() { return !m_pre2icnt_q.size(); }
    void pop_pre2icnt_q() {m_pre2icnt_q.pop_front(); }
    void fill_pre2icnt_q(mem_fetch * mf) { m_pre2icnt_q.push_back(mf); }
    int pre2icnt_q_size() { return m_pre2icnt_q.size(); }

    // jwchoi added for debugging
    void uid_list_in_pre2icnt_q();
	 void addr_list_in_pre2icnt_q();
	 void sid_list_in_pre2icnt_q();
    void wid_list_in_pre2icnt_q();
	 void history_table_list_perwarp();
    void history_table_list_percta_perpc();
    void history_table_list_queue_percta_perpc();
	 void addr_list_in_cal_pref_q();

private:
    //worklist, vertexlist, edgelist, visitedlist, out_worklist. (start, end)
    //EWMA_Unit m_ewma;

    // jyj added : for miss queue and prefetch req queue (1 pref request per 10cycle) - to check 10 cycle
    unsigned long long pre_clock;

    //jwchoi added for config (1122)
    unsigned m_prefetcher_type;
    unsigned ADAPTIVE_ROW_CHANGE;
    unsigned m_knextline;

    unsigned m_cta_x_dim;  // cta dim x
    unsigned m_cta_y_dim;  // cta dim y

    // for each prefetch, add 1 (initial value : -1) --> so first CTA's number = (-1)+1 = 0
    // for each prefetch, calculate : row_change_count + (stride/(4*m_cta_x_dim)) (## multiply 4 because data size is 4 byte)
    // if calculated value == (img_size/cta_x_dim) --> use stirde = current stride + cta_y_dim * img_size * 4 instead of old stride
    // (ex) img size 256, stride is 128(=32), cta is (32,8) --> if cta number(= row_change_count) is 7 : 7+1(=8) > (256/32)-1 (=7) --> row change 
    // For row only now, but will be expanded 
    int row_change_count;
    // row change threshold = (img_size/cta_x_dim)-1
    int row_change_threshold;

    new_addr_type m_start_address;

    unsigned m_img_x_size;  // row size (width)
    unsigned m_img_y_size;  // col size (height)
    unsigned m_img_z_size;  // maybe z siae for 3D image

    unsigned m_prefetch_direction;  // row / col / tensor

    // jyj edited
    // we should save cta id & warp id
    std::list<std::pair<mem_access_t, std::pair<unsigned, unsigned>>> m_req_q;
    //std::list<mem_access_t> m_req_q;   // == cal_pref_q
    std::list<mem_fetch*> m_pre2icnt_q;

    // added by jwchoi
    // todo : warp_inst_t* or mem_access_t*
    // since it needs warp_id and thread_id, setting it as warp_inst_t is reasonable
    // since buffer size is fixed (=HISTORY_BUFFER_ADDR_INDEX_SIZE), don't use vector, use array
    simple_queue_HB prefetch_mem_hb;
    // for perwarp queue : save request per warp
    perwarp_HB prefetch_mem_hb_perwarp;
    // for perCTA, perwarp queue : save request per CTA, warp
    percta_warp_HB prefetch_mem_hb_percta_perwarp;
    // for perCTA, perwarp queue : save request per CTA, pc
    // implementing APRES
    percta_pc_HB prefetch_mem_hb_percta_perpc;

    // jwchoi added 0802 : to implement [inst empty case] : if mf is used for prefetch request generation, remove (pop) it
    percta_pc_HB queue_prefetch_mem_hb_percta_perpc;

    // to record leading warp and intra warp stride per pc;
    // cta id, pc, stride, confidence level
    std::map<unsigned, std::map<unsigned, std::pair<unsigned, unsigned> > > intra_stride_percta_pc;

    pershader_warp_HB prefetch_mem_hb_pershader_perwarp;

    // record pattern
    perpc_pattern_HB prefetch_pattern_hb_percta_perpc;

    //Prefetch_Mode m_mode;
    unsigned m_max_queue_length;
    bool m_double_line;

};


#endif
