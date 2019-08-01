#include <iostream>
#include "prefetcher.h"


//added by jyj
//return next line address
new_addr_type Prefetch_Unit::prefetch_knextline(mem_access_t access){
	//1)make mf with inst or access
	//2)calculate next addr
	//cache line size = 128B
	//sizeof(unsigned long long) = 8B, sizeof(unsigned) = 1B
	new_addr_type next_addr=0;
	new_addr_type stride = 0;
	// row major access
	if(m_prefetch_direction == 1){
		if(ADAPTIVE_ROW_CHANGE == 1){
			if(check_row_change(access.get_addr())){
				// img_x_size *4 
				stride = m_knextline*m_cta_x_dim*4 + (m_cta_y_dim-1)*m_img_x_size*4;
			}
			else{
				stride = m_knextline*m_cta_x_dim*4;
			}
		}
		else{
			stride = m_knextline*m_cta_x_dim*4;
		}
	}
	// column major access
	else if(m_prefetch_direction == 2){
		if(ADAPTIVE_ROW_CHANGE == 1){
			if(check_col_change(access.get_addr())){
				// img_x_size *4 
				stride = m_knextline*m_cta_y_dim*m_img_x_size*4 + (m_cta_x_dim - m_img_x_size*(m_img_y_size - m_cta_y_dim))*4;
			}
			else{
				stride = m_knextline*m_cta_y_dim*m_img_x_size*4;
			}
		}
		else{
			stride = m_knextline*m_cta_y_dim*m_img_x_size*4;
		}
	}
	// change this
	next_addr = access.get_addr() + stride;
	return next_addr;
}



void Prefetch_Unit::do_prefetch(const warp_inst_t &inst, mem_access_t access, unsigned data_size)
{
	new_addr_type addr_tobe_prefetched;

	if(inst.is_load()){
		if(m_prefetcher_type == 0){
			// nextline prefetch
			addr_tobe_prefetched = prefetch_knextline(access);
			if( gen_prefetch_request(addr_tobe_prefetched, inst.get_cta_id(), inst.warp_id(), data_size) ){
			}
		}
		else if(m_prefetcher_type == 1){
		}
	}
}



//added by jyj
//generate mem access with predicted address
//put generated mem access into prefetch queue
// modified by jwchoi (0608)
// return true if request is used for generation prefetch request
bool Prefetch_Unit::gen_prefetch_request(new_addr_type addr, unsigned cta_id, unsigned warp_id, unsigned data_size){
	//std::cout<<"Input addr : "<<addr << std::endl;
	if( is_req_in_queue(addr) ){
		return true;
	}
	else if ( !is_req_in_queue(addr) && !queue_full() ) {
		mem_access_t generated_access(GLOBAL_ACC_R, addr, data_size, false);
		std::pair<mem_access_t, std::pair<unsigned, unsigned>> access_warp = std::make_pair(generated_access, std::make_pair(cta_id, warp_id));
		m_req_q.push_back(access_warp);

		return true;
	}
	else if( !is_req_in_queue(addr) && queue_full() ){
		return false;
	}
}



void Prefetch_Unit::uid_list_in_pre2icnt_q(){
	std::cout << "pre2icnt_q uid list : ";
	std::list<mem_fetch*>::iterator it_preq;
	for(it_preq = m_pre2icnt_q.begin(); it_preq != m_pre2icnt_q.end(); ++it_preq){
		std::cout << (*it_preq)->get_request_uid() <<" / ";
	}
	std::cout<<std::endl;
}

void Prefetch_Unit::addr_list_in_pre2icnt_q(){
	std::cout << "pre2icnt_q addr list : ";
	std::list<mem_fetch*>::iterator it_preq;
	for(it_preq = m_pre2icnt_q.begin(); it_preq != m_pre2icnt_q.end(); ++it_preq){
		std::cout << (*it_preq)->get_addr() <<" / ";
	}
	std::cout<<std::endl;
}

void Prefetch_Unit::sid_list_in_pre2icnt_q(){
	std::cout << "pre2icnt_q sid list : ";
	std::list<mem_fetch*>::iterator it_preq;
	for(it_preq = m_pre2icnt_q.begin(); it_preq != m_pre2icnt_q.end(); ++it_preq){
		std::cout << (*it_preq)->get_sid() <<" / ";
	}
	std::cout<<std::endl;
}

void Prefetch_Unit::wid_list_in_pre2icnt_q(){
	std::cout << "pre2icnt_q wid list : ";
	std::list<mem_fetch*>::iterator it_preq;
	for(it_preq = m_pre2icnt_q.begin(); it_preq != m_pre2icnt_q.end(); ++it_preq){
		std::cout << (*it_preq)->get_wid() <<" / ";
	}
	std::cout<<std::endl;
}


void Prefetch_Unit::history_table_list_perwarp(){
	std::cout << "----- Histroy table list -----"<<std::endl;
	std::map<unsigned, std::deque<prefetcher_mem_hb_t> >::iterator it_history;
	for(it_history = prefetch_mem_hb_perwarp.begin(); it_history != prefetch_mem_hb_perwarp.end(); ++it_history){
		std::cout<<"warp id : "<<it_history->first<<std::endl;
		std::deque<prefetcher_mem_hb_t>::iterator it_perwarp;
		for(it_perwarp = (it_history->second).begin(); it_perwarp != (it_history->second).end(); ++it_perwarp){
			std::cout<<"pc : "<<(*it_perwarp).pc<<" / ";
			std::cout<<"addr : "<<(*it_perwarp).addr<<" / ";
			std::cout<<"sid : "<<(*it_perwarp).shader_id<<" / ";
			std::cout<<"wid : "<<(*it_perwarp).warp_id<<" / ";
			std::cout<<"stride : "<<(*it_perwarp).stride<<std::endl;
		}
	}
	std::cout << "------------------------------"<<std::endl;
}

void Prefetch_Unit::history_table_list_percta_perpc(){
	std::cout << "----- Histroy table list : percta & per pc -----"<<std::endl;
	std::map<unsigned, std::map<unsigned, std::deque<prefetcher_mem_hb_t> > >::iterator it_history;
	for(it_history = prefetch_mem_hb_percta_perpc.begin(); it_history != prefetch_mem_hb_percta_perpc.end(); ++it_history){
		std::cout<<"cta id : "<<it_history->first<<std::endl;
		std::map<unsigned, std::deque<prefetcher_mem_hb_t> >::iterator it_perpc;
		for(it_perpc = (it_history->second).begin(); it_perpc != (it_history->second).end(); ++it_perpc){
			std::cout<<"pc : "<<it_perpc->first<<std::endl;
			std::deque<prefetcher_mem_hb_t>::iterator it_perpc_warps;
			for(it_perpc_warps = (it_perpc->second).begin(); it_perpc_warps != (it_perpc->second).end(); ++it_perpc_warps){
				std::cout<<"pc : "<<(*it_perpc_warps).pc<<" / ";
				std::cout<<"addr : "<<(*it_perpc_warps).addr<<" / ";
				std::cout<<"sid : "<<(*it_perpc_warps).shader_id<<" / ";
				std::cout<<"wid : "<<(*it_perpc_warps).warp_id<<" / ";
				//std::cout<<"stride : "<<(*it_perpc_warps).stride<<std::endl;
				std::cout<<"stride : "<<(*it_perpc_warps).stride<<" / ";
				std::cout<<"inst uid : "<<(*it_perpc_warps).inst_uid<<" / ";
				std::cout<<"is used : "<<(*it_perpc_warps).used_for_prefetch_request<<" / ";
				std::cout<<"timestamp : "<<(*it_perpc_warps).timestamp<<std::endl;
			}
		}
	}
	std::cout << "------------------------------"<<std::endl;
}

void Prefetch_Unit::history_table_list_queue_percta_perpc(){
	std::cout << "----- Histroy table list : percta & per pc -----"<<std::endl;
	std::map<unsigned, std::map<unsigned, std::deque<prefetcher_mem_hb_t> > >::iterator it_history;
	for(it_history = queue_prefetch_mem_hb_percta_perpc.begin(); it_history != queue_prefetch_mem_hb_percta_perpc.end(); ++it_history){
		std::cout<<"cta id : "<<it_history->first<<std::endl;
		std::map<unsigned, std::deque<prefetcher_mem_hb_t> >::iterator it_perpc;
		for(it_perpc = (it_history->second).begin(); it_perpc != (it_history->second).end(); ++it_perpc){
			std::cout<<"pc : "<<it_perpc->first<<std::endl;
			std::deque<prefetcher_mem_hb_t>::iterator it_perpc_warps;
			for(it_perpc_warps = (it_perpc->second).begin(); it_perpc_warps != (it_perpc->second).end(); ++it_perpc_warps){
				std::cout<<"pc : "<<(*it_perpc_warps).pc<<" / ";
				std::cout<<"addr : "<<(*it_perpc_warps).addr<<" / ";
				std::cout<<"sid : "<<(*it_perpc_warps).shader_id<<" / ";
				std::cout<<"wid : "<<(*it_perpc_warps).warp_id<<" / ";
				//std::cout<<"stride : "<<(*it_perpc_warps).stride<<std::endl;
				std::cout<<"stride : "<<(*it_perpc_warps).stride<<" / ";
				std::cout<<"inst uid : "<<(*it_perpc_warps).inst_uid<<std::endl;
			}
		}
	}
	std::cout << "------------------------------"<<std::endl;
}

void Prefetch_Unit::addr_list_in_cal_pref_q(){
	std::cout << "cal_pref_q addr list : ";
	std::list<std::pair<mem_access_t, std::pair<unsigned, unsigned>>>::iterator it_preq;
	for(it_preq = m_req_q.begin(); it_preq != m_req_q.end(); ++it_preq){
		std::cout << ((*it_preq).first).get_addr() <<" / ";
	}
	std::cout<<std::endl;
}


