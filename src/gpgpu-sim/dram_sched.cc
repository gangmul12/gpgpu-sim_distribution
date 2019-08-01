// Copyright (c) 2009-2011, Tor M. Aamodt, Ali Bakhoda, George L. Yuan,
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

#include "dram_sched.h"
#include "gpu-misc.h"
#include "gpu-sim.h"
#include "../abstract_hardware_model.h"
#include "mem_latency_stat.h"
#include "stat-tool.h"
frfcfs_scheduler::frfcfs_scheduler( const memory_config *config, dram_t *dm, memory_stats_t *stats )
{
   m_config = config;
   m_stats = stats;
   m_num_pending = 0;
   m_num_write_pending = 0;
   m_dram = dm;
   m_queue = new std::list<dram_req_t*>[m_config->nbk];
   m_bins = new std::map<unsigned,std::list<std::list<dram_req_t*>::iterator> >[ m_config->nbk ];
   m_last_row = new std::list<std::list<dram_req_t*>::iterator>*[ m_config->nbk ];
   curr_row_service_time = new unsigned[m_config->nbk];
   row_service_timestamp = new unsigned[m_config->nbk];
   for ( unsigned i=0; i < m_config->nbk; i++ ) {
      m_queue[i].clear();
      m_bins[i].clear();
      m_last_row[i] = NULL;
      curr_row_service_time[i] = 0;
      row_service_timestamp[i] = 0;
   }
   if(m_config->seperate_write_queue_enabled) {
	   m_write_queue = new std::list<dram_req_t*>[m_config->nbk];
	   m_write_bins = new std::map<unsigned,std::list<std::list<dram_req_t*>::iterator> >[ m_config->nbk ];
	   m_last_write_row = new std::list<std::list<dram_req_t*>::iterator>*[ m_config->nbk ];

	   for ( unsigned i=0; i < m_config->nbk; i++ ) {
	         m_write_queue[i].clear();
	         m_write_bins[i].clear();
	         m_last_write_row[i] = NULL;
	      }
   }
   m_mode = READ_MODE;

}

void frfcfs_scheduler::add_req( dram_req_t *req )
{
  if(m_config->seperate_write_queue_enabled && req->data->is_write()) {
	  assert(m_num_write_pending < m_config->gpgpu_frfcfs_dram_write_queue_size);
	  m_num_write_pending++;
	  m_write_queue[req->bk].push_front(req);
	  std::list<dram_req_t*>::iterator ptr = m_write_queue[req->bk].begin();
	  m_write_bins[req->bk][req->row].push_front( ptr ); //newest reqs to the front
  } else {
	   assert(m_num_pending < m_config->gpgpu_frfcfs_dram_sched_queue_size);
	   m_num_pending++;
	   m_queue[req->bk].push_front(req);
	   std::list<dram_req_t*>::iterator ptr = m_queue[req->bk].begin();
	   m_bins[req->bk][req->row].push_front( ptr ); //newest reqs to the front
  }
}

void frfcfs_scheduler::data_collection(unsigned int bank)
{
   if (gpu_sim_cycle > row_service_timestamp[bank]) {
      curr_row_service_time[bank] = gpu_sim_cycle - row_service_timestamp[bank];
      if (curr_row_service_time[bank] > m_stats->max_servicetime2samerow[m_dram->id][bank])
         m_stats->max_servicetime2samerow[m_dram->id][bank] = curr_row_service_time[bank];
   }
   curr_row_service_time[bank] = 0;
   row_service_timestamp[bank] = gpu_sim_cycle;
   if (m_stats->concurrent_row_access[m_dram->id][bank] > m_stats->max_conc_access2samerow[m_dram->id][bank]) {
      m_stats->max_conc_access2samerow[m_dram->id][bank] = m_stats->concurrent_row_access[m_dram->id][bank];
   }
   m_stats->concurrent_row_access[m_dram->id][bank] = 0;
   m_stats->num_activates[m_dram->id][bank]++;
}

dram_req_t *frfcfs_scheduler::schedule( unsigned bank, unsigned curr_row )
{
   //row
   bool rowhit = true;
   std::list<dram_req_t*> *m_current_queue = m_queue;
   std::map<unsigned,std::list<std::list<dram_req_t*>::iterator> > *m_current_bins = m_bins ;
   std::list<std::list<dram_req_t*>::iterator> **m_current_last_row = m_last_row;

   if(m_config->seperate_write_queue_enabled) {
	   if(m_mode == READ_MODE &&
			  ((m_num_write_pending >= m_config->write_high_watermark )
			  // || (m_queue[bank].empty() && !m_write_queue[bank].empty())
			   )) {
		   m_mode = WRITE_MODE;
	   }
	   else if(m_mode == WRITE_MODE &&
				  (( m_num_write_pending < m_config->write_low_watermark )
				 //  || (!m_queue[bank].empty() && m_write_queue[bank].empty())
				   )){
		   m_mode = READ_MODE;
	   }
   }

   if(m_mode == WRITE_MODE) {
	   m_current_queue = m_write_queue;
	   m_current_bins = m_write_bins ;
	   m_current_last_row = m_last_write_row;
   }

   if ( m_current_last_row[bank] == NULL ) {
      if ( m_current_queue[bank].empty() )
         return NULL;

      std::map<unsigned,std::list<std::list<dram_req_t*>::iterator> >::iterator bin_ptr = m_current_bins[bank].find( curr_row );
      if ( bin_ptr == m_current_bins[bank].end()) {
         dram_req_t *req = m_current_queue[bank].back();
         bin_ptr = m_current_bins[bank].find( req->row );
         assert( bin_ptr != m_current_bins[bank].end() ); // where did the request go???
         m_current_last_row[bank] = &(bin_ptr->second);
         data_collection(bank);
         rowhit = false;
      } else {
    	  m_current_last_row[bank] = &(bin_ptr->second);
         rowhit = true;
      }
   }
   std::list<dram_req_t*>::iterator next = m_current_last_row[bank]->back();
   dram_req_t *req = (*next);

   //rowblp stats
    m_dram->access_num++;
    bool is_write = req->data->is_write();
    if(is_write)
  	  m_dram->write_num++;
    else
  	  m_dram->read_num++;

    if(rowhit) {
     m_dram->hits_num++;
     if(is_write)
    	  m_dram->hits_write_num++;
      else
    	  m_dram->hits_read_num++;
    }

   m_stats->concurrent_row_access[m_dram->id][bank]++;
   m_stats->row_access[m_dram->id][bank]++;
   m_current_last_row[bank]->pop_back();

   m_current_queue[bank].erase(next);
   if ( m_current_last_row[bank]->empty() ) {
	   m_current_bins[bank].erase( req->row );
	   m_current_last_row[bank] = NULL;
   }
#ifdef DEBUG_FAST_IDEAL_SCHED
   if ( req )
      printf("%08u : DRAM(%u) scheduling memory request to bank=%u, row=%u\n", 
             (unsigned)gpu_sim_cycle, m_dram->id, req->bk, req->row );
#endif

   if(m_config->seperate_write_queue_enabled && req->data->is_write()) {
	   assert( req != NULL && m_num_write_pending != 0 );
	   m_num_write_pending--;
   }
   else {
	   assert( req != NULL && m_num_pending != 0 );
	   m_num_pending--;
   }

   return req;
}


void frfcfs_scheduler::print( FILE *fp )
{
   for ( unsigned b=0; b < m_config->nbk; b++ ) {
      printf(" %u: queue length = %u\n", b, (unsigned)m_queue[b].size() );
   }
}

void dram_t::scheduler_frfcfs()
{
   unsigned mrq_latency;
   frfcfs_scheduler *sched = m_frfcfs_scheduler;
   while ( !mrqq->empty() ) {
      dram_req_t *req = mrqq->pop();

      // Power stats
      //if(req->data->get_type() != READ_REPLY && req->data->get_type() != WRITE_ACK)
      m_stats->total_n_access++;

      if(req->data->get_type() == WRITE_REQUEST){
    	  m_stats->total_n_writes++;
      }else if(req->data->get_type() == READ_REQUEST){
    	  m_stats->total_n_reads++;
      }

      req->data->set_status(IN_PARTITION_MC_INPUT_QUEUE,gpu_sim_cycle+gpu_tot_sim_cycle);
      sched->add_req(req);
   }

   dram_req_t *req;
   unsigned i;
   for ( i=0; i < m_config->nbk; i++ ) {
      unsigned b = (i+prio)%m_config->nbk;
      if ( !bk[b]->mrq ) {

         req = sched->schedule(b, bk[b]->curr_row);

         if ( req ) {
            req->data->set_status(IN_PARTITION_MC_BANK_ARB_QUEUE,gpu_sim_cycle+gpu_tot_sim_cycle);
            prio = (prio+1)%m_config->nbk;
            bk[b]->mrq = req;
            if (m_config->gpgpu_memlatency_stat) {
               mrq_latency = gpu_sim_cycle + gpu_tot_sim_cycle - bk[b]->mrq->timestamp;
               m_stats->tot_mrq_latency += mrq_latency;
               m_stats->tot_mrq_num++;
               bk[b]->mrq->timestamp = gpu_tot_sim_cycle + gpu_sim_cycle;
               m_stats->mrq_lat_table[LOGB2(mrq_latency)]++;
               if (mrq_latency > m_stats->max_mrq_latency) {
                  m_stats->max_mrq_latency = mrq_latency;
               }
            }

            break;
         }
      }
   }
}



approx_scheduler::approx_scheduler(const memory_config *config, dram_t *dm, memory_stats_t *stats) {
   m_config = config;
   m_stats = stats;
   m_num_pending = 0;
   m_dram = dm;
	m_row_sense = 0;
	m_nrow_per_bank_for_full = 32 / m_config->nbk; //maybe two for HBM
	m_precision = m_config -> gpgpu_approx_dram_precision;

	//parse Transpose unit target 
	//char* tu_target = m_config -> gpgpu_approx_dram_tu_location;
	//if(strcmp(tu_target, "L1") == 0){
	//	std::cout<<"tutarget L1"<<std::endl;
	//	m_tu_target=L1;
	//}
	//else if(strcmp(tu_target, "L2")==0)
	//	m_tu_target=L2;
	//else
	//	assert(0&&"tu target error. should be L1 or L2");
	//char* line_str = (m_tu_target == L1)? m_config->m_l1d_str : m_config->m_L2_config.m_config_string;	
	//assert(line_str != NULL &&"registering tu target failed.");
	//sscanf(line_str, "%u:%u:", &m_linesize, &m_linesize);
	m_linesize=1024;
	assert(m_linesize==1024&&"1024 maybe. other values are very strange");
   m_queue = new std::list<dram_req_t*>;
	m_queue->clear();
	for(unsigned i = 0 ; i < 2 ; i++){
		m_current_req[i]=NULL;
		m_flag_subreq_done[i].assign(m_nrow_per_bank_for_full*m_config->nbk, false);
	}

}
void approx_scheduler::add_req( dram_req_t *req )
{
	//req->data->print(stdout);
   if(req->data->is_approx() && req->data->get_type()==READ_REQUEST){
		assert(req->data->get_data_size()==m_linesize);
		unsigned precision = m_precision;
		//TODO should be separate function?
		unsigned row_iter = (precision%m_config->nbk) ? precision / m_config->nbk + 1 : precision/m_config->nbk;
		for(unsigned io = 0 ; io < row_iter; io++, m_row_sense++)
		{
			unsigned bank_iter = (io==row_iter-1) ? (precision % m_config->nbk) : m_config->nbk;
			if(bank_iter==0) bank_iter = m_config->nbk;
			m_row_sense = m_row_sense % (row_iter);
			for(unsigned i = 0 ; i < bank_iter ; i++){
				dram_req_t* sub_req = new dram_req_t(req->data, m_config->nbk, m_config->dram_bnk_indexing_policy);
				sub_req->bk=i;
				sub_req->row += m_row_sense;
				sub_req->nbytes = 32;
				m_queue->push_front(sub_req);
				// stats
				unsigned dram_id = req->data->get_tlx_addr().chip;
				if (m_config->gpgpu_memlatency_stat) { 
      			if (req->data->get_is_write()) {
         			if ( req->data->get_access_type() != L2_WRBK_ACC ) {   //do not count L2_writebacks here 
            			m_stats->bankwrites[req->data->get_sid()][dram_id][sub_req->bk]++;
            			shader_mem_acc_log( req->data->get_sid(), dram_id, sub_req->bk, 'w');
         			}
         			m_stats->totalbankwrites[dram_id][sub_req->bk]++;
      			} else {
         			m_stats->bankreads[req->data->get_sid()][dram_id][sub_req->bk]++;
         			shader_mem_acc_log( req->data->get_sid(), dram_id, sub_req->bk, 'r');
         			m_stats->totalbankreads[dram_id][sub_req->bk]++;
      			}
      			m_stats->mem_access_type_stats[req->data->get_access_type()][dram_id][sub_req->bk]++;
   			}
			}
		}

		m_num_pending+=precision;
		unsigned slot_idx=0;
		if(m_current_req[0]==NULL){
			slot_idx=0;
		}
		else{
			if(m_current_req[1]!=NULL){
				printf("num pending : %u\n", m_num_pending);
				m_current_req[1]->data->print(stdout);
				assert((m_current_req[1]==NULL)&&"approx scheduler double req buffer screwed");
			}
			slot_idx = 1;
		}
		
		m_current_req[slot_idx]=req;
		m_flag_subreq_done[slot_idx].assign(precision, false);
	}
	else{
		unsigned dram_id = req->data->get_tlx_addr().chip;
		if (m_config->gpgpu_memlatency_stat) { 
  			if (req->data->get_is_write()) {
     			if ( req->data->get_access_type() != L2_WRBK_ACC ) {   //do not count L2_writebacks here 
        			m_stats->bankwrites[req->data->get_sid()][dram_id][req->bk]++;
        			shader_mem_acc_log( req->data->get_sid(), dram_id, req->bk, 'w');
     			}
     			m_stats->totalbankwrites[dram_id][req->bk]++;
  			} else {
     			m_stats->bankreads[req->data->get_sid()][dram_id][req->bk]++;
     			shader_mem_acc_log( req->data->get_sid(), dram_id, req->bk, 'r');
     			m_stats->totalbankreads[dram_id][req->bk]++;
  			}
  			m_stats->mem_access_type_stats[req->data->get_access_type()][dram_id][req->bk]++;
		}
		m_num_pending++;
	  	m_queue->push_front(req);
	}
}
dram_req_t* approx_scheduler::schedule(unsigned bank, unsigned curr_row){



	assert(m_num_pending>0 && !m_queue->empty());
	
	m_num_pending--;
	bool rowhit = true;
	dram_req_t* result = m_queue->back();
	assert(result->bk == bank);
	m_queue->pop_back();
	if(curr_row == result->row){
		
	}
	else{
		m_stats->concurrent_row_access[m_dram->id][bank] = 0;
   	m_stats->num_activates[m_dram->id][bank]++;
		rowhit = false;
	}
	m_stats->concurrent_row_access[m_dram->id][bank]++;

   //rowblp stats
   m_dram->access_num++;
   bool is_write = result->data->is_write();
   if(is_write)
  		m_dram->write_num++;
   else
		m_dram->read_num++;
 	
	if(rowhit) {
     m_dram->hits_num++;
     if(is_write)
    	  m_dram->hits_write_num++;
     else
    	  m_dram->hits_read_num++;
   }
   m_stats->row_access[m_dram->id][result->bk]++;

	return result;

}
dram_req_t* approx_scheduler::next_mrq(){
	if(!m_queue->empty())
		return m_queue->back();
	else
		return NULL;
}
void dram_t::scheduler_approx(){
	approx_scheduler *sched = m_approx_scheduler;
	while ( !mrqq->empty() && sched->slot_available()) {
		dram_req_t *req = mrqq->pop();

		// Power stats
		//if(req->data->get_type() != READ_REPLY && req->data->get_type() != WRITE_ACK)
		m_stats->total_n_access++;
		if(req->data->get_type() == WRITE_REQUEST){
			m_stats->total_n_writes++;
		}else if(req->data->get_type() == READ_REQUEST){
			m_stats->total_n_reads++;
		}
		req->data->set_status(IN_PARTITION_MC_INPUT_QUEUE,gpu_sim_cycle+gpu_tot_sim_cycle);
		sched->add_req(req);
	}
	for(unsigned jj = 0 ; jj < 1&& sched->num_pending() ; jj++){
	dram_req_t *req;
	
	unsigned b=-1;
	if(sched->num_pending()){
		b = (sched->next_mrq())->bk;
		assert(b>=0);
		if(!bk[b]->mrq){
			req = sched->schedule(b, bk[b]->curr_row);
			bk[b]->mrq = req;
			if (m_config->gpgpu_memlatency_stat) {
				unsigned mrq_latency = gpu_sim_cycle + gpu_tot_sim_cycle - bk[b]->mrq->timestamp;
            m_stats->tot_mrq_latency += mrq_latency;
            m_stats->tot_mrq_num++;
            bk[b]->mrq->timestamp = gpu_tot_sim_cycle + gpu_sim_cycle;
            m_stats->mrq_lat_table[LOGB2(mrq_latency)]++;
            if (mrq_latency > m_stats->max_mrq_latency) {
               m_stats->max_mrq_latency = mrq_latency;
				}
         }

			if(req->data->get_status() != IN_PARTITION_MC_BANK_ARB_QUEUE)
				req->data->set_status(IN_PARTITION_MC_BANK_ARB_QUEUE, gpu_sim_cycle+gpu_tot_sim_cycle);
		}
	}
	}

}
mem_fetch* approx_scheduler::process_return_cmd(dram_req_t* cmd){
	
	if((!cmd->data->is_approx()) || cmd->data->is_write()){

		return cmd->data;
	}
	else{
		unsigned idx = -1; //idx for double buffer
		if(m_current_req[0]){
			
			idx = (cmd->data == m_current_req[0]->data)? 0 : 1;
			if(idx){

				assert(cmd->data == m_current_req[1]->data);
			}
		}
		else{
			idx = 1;

			assert(m_current_req[1]!=NULL);
		}
		unsigned row = cmd->row - m_current_req[idx]->row;
		unsigned bk = cmd->bk;
		unsigned subidx = row*m_config->nbk + bk; // idx for sub request
		unsigned precision = m_precision;
		assert(m_flag_subreq_done[idx][subidx]==false);
		m_flag_subreq_done[idx][subidx]=true;
		bool done=true;
		for(unsigned i = 0; i < precision ; i++){
			if(m_flag_subreq_done[idx][i]==false){
				done=false;
				break;
			}
		}
		if(done){
			m_current_req[idx]=NULL;
			return cmd->data;
		}
		else
			return NULL;
	}
	return NULL;
}
unsigned approx_scheduler::num_pending() const{
//	unsigned result=0;
//	if(m_current_req[0]) result++;
//	if(m_current_req[1]) result++;
	return m_num_pending;
}

bool approx_scheduler::slot_available() const{
	bool result = (m_current_req[0]==NULL || m_current_req[1]==NULL);
	return result;
}
