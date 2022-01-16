Search.setIndex({docnames:["api/graph2tensor.client","api/graph2tensor.common","api/graph2tensor.converter","api/graph2tensor.egograph","api/graph2tensor.interface","api/graph2tensor.model","api/graph2tensor.sampler","api/serving","index"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":4,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,sphinx:56},filenames:["api/graph2tensor.client.rst","api/graph2tensor.common.rst","api/graph2tensor.converter.rst","api/graph2tensor.egograph.rst","api/graph2tensor.interface.rst","api/graph2tensor.model.rst","api/graph2tensor.sampler.rst","api/serving.rst","index.rst"],objects:{"graph2tensor.client":{NumpyGraph:[0,1,1,""]},"graph2tensor.client.NumpyGraph":{add_edge:[0,2,1,""],add_node:[0,2,1,""],from_config:[0,2,1,""],get_dst_type:[0,2,1,""],get_edge_attr_data:[0,2,1,""],get_edge_attr_info:[0,2,1,""],get_edge_ids:[0,2,1,""],get_edge_probs:[0,2,1,""],get_node_attr_data:[0,2,1,""],get_node_attr_info:[0,2,1,""],get_node_label:[0,2,1,""],get_src_type:[0,2,1,""],is_edge_directed:[0,2,1,""],is_node_labeled:[0,2,1,""],lookup_edges:[0,2,1,""],lookup_nodes:[0,2,1,""],node2vec_walk:[0,2,1,""],random_walk:[0,2,1,""],sample_neighbors:[0,2,1,""],schema:[0,3,1,""],to_config:[0,2,1,""]},"graph2tensor.client.distributed":{DistNumpyAttributeStoreServicer:[0,1,1,""],DistNumpyGraph:[0,1,1,""],DistNumpyRelationStoreServicer:[0,1,1,""]},"graph2tensor.client.distributed.DistNumpyAttributeStoreServicer":{add_attr:[0,2,1,""],add_label:[0,2,1,""]},"graph2tensor.client.distributed.DistNumpyGraph":{add_edge:[0,2,1,""],add_node:[0,2,1,""],from_config:[0,2,1,""],get_dst_type:[0,2,1,""],get_edge_attr_data:[0,2,1,""],get_edge_attr_info:[0,2,1,""],get_edge_ids:[0,2,1,""],get_edge_probs:[0,2,1,""],get_node_attr_data:[0,2,1,""],get_node_attr_info:[0,2,1,""],get_node_label:[0,2,1,""],get_src_type:[0,2,1,""],is_edge_directed:[0,2,1,""],is_node_labeled:[0,2,1,""],lookup_edges:[0,2,1,""],lookup_nodes:[0,2,1,""],node2vec_walk:[0,2,1,""],random_walk:[0,2,1,""],sample_neighbors:[0,2,1,""],schema:[0,3,1,""],to_config:[0,2,1,""]},"graph2tensor.common":{Edges:[1,1,1,""],Nodes:[1,1,1,""]},"graph2tensor.common.Edges":{dst_ids:[1,3,1,""],edge_ids:[1,3,1,""],edge_type:[1,3,1,""],src_ids:[1,3,1,""]},"graph2tensor.common.Nodes":{ids:[1,3,1,""],node_type:[1,3,1,""],offset:[1,3,1,""]},"graph2tensor.converter":{Ego2Tensor:[2,1,1,""],SkipGram:[2,1,1,""]},"graph2tensor.converter.Ego2Tensor":{convert:[2,2,1,""]},"graph2tensor.converter.SkipGram":{convert:[2,2,1,""]},"graph2tensor.egograph":{EgoGraph:[3,1,1,""]},"graph2tensor.egograph.EgoGraph":{add_path:[3,2,1,""],centre_nodes:[3,3,1,""],paths:[3,3,1,""]},"graph2tensor.interface":{build_model:[4,0,1,""],evaluate:[4,0,1,""],graph_from_dgl:[4,0,1,""],graph_from_pg:[4,0,1,""],graph_to_dgl:[4,0,1,""],predict:[4,0,1,""],predict_and_explain:[4,0,1,""],train:[4,0,1,""]},"graph2tensor.model.data":{EgoTensorGenerator:[5,1,1,""],SkipGramGenerator4DeepWalk:[5,1,1,""],SkipGramGenerator4MetaPath2Vec:[5,1,1,""],SkipGramGenerator4Node2Vec:[5,1,1,""]},"graph2tensor.model.explainer":{IntegratedGradients:[5,1,1,""]},"graph2tensor.model.explainer.IntegratedGradients":{explain:[5,2,1,""]},"graph2tensor.model.layers":{AttrCompact:[5,1,1,""],EmbeddingEncoder:[5,1,1,""],GATConv:[5,1,1,""],GCNConv:[5,1,1,""],GINConv:[5,1,1,""],IntegerLookupEncoder:[5,1,1,""],OnehotEncoder:[5,1,1,""],RGCNConv:[5,1,1,""],StringLookupEncoder:[5,1,1,""],UniMP:[5,1,1,""]},"graph2tensor.model.models":{DeepWalk:[5,1,1,""],MessagePassing:[5,1,1,""],MetaPath2Vec:[5,1,1,""],Node2Vec:[5,1,1,""]},"graph2tensor.model.models.DeepWalk":{get_node_embedding:[5,2,1,""],most_similar:[5,2,1,""]},"graph2tensor.model.models.MetaPath2Vec":{get_node_embedding:[5,2,1,""],most_similar:[5,2,1,""]},"graph2tensor.model.models.Node2Vec":{get_node_embedding:[5,2,1,""],most_similar:[5,2,1,""]},"graph2tensor.sampler":{MetaPathRandomWalker:[6,1,1,""],MetaPathSampler:[6,1,1,""],Node2VecWalker:[6,1,1,""],RandomWalker:[6,1,1,""]},"graph2tensor.sampler.MetaPathRandomWalker":{sample:[6,2,1,""]},"graph2tensor.sampler.MetaPathSampler":{sample:[6,2,1,""]},"graph2tensor.sampler.Node2VecWalker":{sample:[6,2,1,""]},"graph2tensor.sampler.RandomWalker":{sample:[6,2,1,""]},app:{predict:[7,0,1,""],predict_and_explain:[7,0,1,""],train:[7,0,1,""]}},objnames:{"0":["py","function","Python function"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","property","Python property"]},objtypes:{"0":"py:function","1":"py:class","2":"py:method","3":"py:property"},terms:{"0":[0,1,2,4,5,6,7],"0001":[],"001":[],"00653":[],"025":[],"04212":[],"05":[5,6],"09":7,"1":[1,2,4,5,6,7],"10":[1,4,5,6,7],"100":[],"10000":[],"1024":[],"10th":[],"11":1,"12":1,"127":4,"128":[4,7],"13":1,"14":1,"15":[1,7],"15432":4,"16":1,"1607":[],"169343":5,"17":1,"18":1,"1804":[],"19":1,"1971":[4,7],"1986":[4,7],"1987":[4,7],"1988":[4,7],"1990":[4,7],"1d":[],"1e":[5,6],"1gb":[],"1st":[4,5],"2":[1,2,4,5],"20":1,"2021":[4,7],"21":7,"25":4,"256":4,"2d":[],"2nd":[4,5],"3":[1,2,5,6],"30":[],"32":[4,5,7],"33":[],"37":[4,7],"4":[0,1,5],"40":[4,7],"4096":0,"40960":5,"5":[1,2,5],"55":[],"6":1,"64":[4,7],"67":[],"7":1,"75":[],"8":[1,5],"9":[1,4],"abstract":[1,3],"boolean":[],"case":[],"class":[4,7,8],"default":[0,1,5],"do":[],"dupr\u00e9":[],"final":[],"float":[0,4],"function":[],"import":[0,1,7],"int":[0,1,4,5,6],"long":[],"new":[0,3],"return":[0,1,2,3,4,5,6,7],"static":[],"super":5,"throw":[],"true":[0,2,4,5,6,7],"try":[],"while":[],A:[0,2,5,6],As:[],At:[],By:[],For:[0,5],If:6,In:[0,1],It:6,Its:[],NOT:[],The:[0,1,3,6],These:[],To:6,_0:5,_:5,__call__:[],__init__:[],_i:5,_icup:[],_r:5,a_:5,a_out:[],ab:[],abc:[],abl:[],about:[],abov:0,absolut:[],acc:[],accept:[],access:[],accord:0,account:[],accumul:[],accuraci:[],across:[],activ:[4,5,7],activity_regular:[],actual:[],ad:[0,3],adam:[4,7],adapt:[],add:[0,3,5],add_attr:0,add_edg:0,add_label:0,add_loss:[],add_metr:[],add_nod:0,add_path:3,add_self_loop:[4,5,7],add_upd:[],add_vari:[],add_weight:[],addit:[],adher:[],adjac:6,advis:[],affect:[],after:5,ag:0,aggr:5,aggr_typ:5,aggreg:5,agnost:[],algorithm:[],alia:[],align:5,all:[0,1,3,4,6],allow:[],along:[0,5,6],alpha:[],alpha_:5,alreadi:[],also:[0,4,5],ambigu:[],amount:[],an:[0,2,5,6],ani:[0,4],anoth:[],apart:[],api:[],api_doc:[],app:7,appli:[0,5,6],applic:[],appropri:[],ar:[0,1,5],arang:[1,5],architectur:7,aren:[],arg:[0,4,5],argument:[],arrai:[0,1],arxiv:[],assert:[],assign:[],associ:[],assum:[],attach:[],attempt:[],attent:5,attr1:4,attr2:4,attr3:4,attr:[0,4],attr_data:0,attr_nam:0,attr_reduce_mod:[4,5],attr_typ:0,attrconcat:[4,7],attribut:[0,2,4,5],attribute_target:0,attributeerror:[],attrs_info:0,auto:[],automat:[],avail:5,averag:[4,5],avg:5,awai:[],b:5,b_:5,b_out:[],back:[],backpropag:[],backward:[],bar:[],base:[0,5],basegraph:0,baselin:5,basi:5,basic:7,batch:[0,1,3],batch_siz:[0,4],batch_word:[],batchnorm:[],becaus:[],becom:[],been:[],befor:[],begin:5,behav:[],behavior:[],being:1,belong:1,below:[4,7],best:[],beta_:5,better:[],between:5,bfloat16:[],bia:5,bias:[],bigoplu:5,bigoplus_:5,binaryaccuraci:[],block:5,bool:[0,4],boost:[],both:1,bottom:[],br:5,breadth:[],browncorpu:[],build:4,build_vocab:[],built:[],by_nam:[],c:5,cach:[],caching_devic:[],calcul:[],call:[],callabl:5,callback:[],callbackany2vec:[],callbacklist:[],caller:[],can:[0,5],candid:[],cannot:[],capabl:[],captur:[],card:[],casel:[],cast:[],cat:[],cate_feat_def:[4,5,7],categor:[],categoricalaccuraci:[],categoryencod:5,caus:[],cbow:[],cbow_mean:[],cdot:5,ce:[],centr:[3,5,6],centre_nod:3,certain:[],chang:5,character:1,check:0,checkpoint:4,checkpoint_dir:4,checkpoint_save_freq:[4,7],checkpointopt:[],children:[],choos:[],chosen:[],chunk:[],cite:[4,5],cl:0,class_weight:[],classmethod:0,clear:[],client:[0,4,6],code:[],coeffici:[],collect:[],common:[0,1,3,6],compact:5,compat:[],compil:[],compos:[],composit:5,comput:0,compute_dtyp:[],compute_loss:[],compute_mask:[],compute_output_shap:[],compute_output_signatur:[],concat:[4,5],concat_hidden:[4,5],concat_target_context:5,concaten:[4,5],config:0,configur:[],conjunct:[],connect:5,consid:[],consist:[5,6],constant:[],constant_initi:[],constraint:[],construct:[],constructor:[],consum:[],contain:5,context:[2,5],contribut:[],control:[],conv2d:[],conv_lay:5,conv_layers_def:[4,7],convers:[],convert:[5,8],convert_to_tensor:[],converter_process_num:[4,5],convolut:[4,5],copi:[],corpora:[],corpu:[],corpus_fil:[],correspond:5,cosin:5,could:[0,1,2,5],count:[],count_param:[],creat:8,crossentropi:[],cup:5,current:[],custom:[],custom_object:[],cython:[],d0:[],d:[2,5,6],data:[0,4,7,8],data_gen:5,databas:4,datagener:4,dataset:[4,5],dataset_def:4,datasetcr:[],datatyp:[],date:7,date_end:[],date_start:[],db_databas:4,db_host:4,db_password:4,db_port:4,db_user:4,deal:[],debug:[],declar:[],decor:[],decreas:[],deepwalk:[],def:[],default_float_attr:4,default_int_attr:4,default_str_attr:4,defin:[5,6],definit:[4,7],del:[],deleg:[],delet:[],deliv:[],dens:[4,7],depend:[],deprec:[],dept:0,descend:[],descent:[],describ:[0,6],descript:7,deseri:0,design:[],destin:[0,1],detail:[0,4,5,6,7],determin:[],determinist:[],dgl:4,dgl_graph:4,dglgraph:4,dglheterograph:4,diagon:5,dict:[0,2,4,5],dictionari:[],differ:5,dimens:5,dimension:[],direct:[0,4],directli:[],directori:4,disabl:[],discard:[0,5,6],discard_frequent_nod:[0,5,6],discuss:[],disk:[],displai:[],distanc:[],distinct:5,distnumpyattributestoreservic:8,distnumpygraph:8,distnumpyrelationstoreservic:8,distribut:[0,2,5,6],distribute_strategi:[],distributionstrategi:[],divid:5,dn:[],doc:[],document:[],doe:[],doesn:[],dog:[],don:[],done:[],downsampl:[],draw:[],drawn:[],drop:[],dropout:[],ds:5,dst:[0,1,2,5,6],dst_id:[0,1,2],dst_type:[0,4],dtype:5,dtype_polici:[],dump:[0,7],dure:[],dynam:[],e1:6,e2:6,e3:6,e:[0,2,5,6,7],e_:5,each:[0,2,3,4,5,6],eager:[],eagerli:[],easier:[],easili:[],edg:[0,2,3,4,5,6,8],edge_attr:0,edge_id:[0,1],edge_import:[],edge_prob:[0,4],edge_typ:[0,1,5,6],edgea:4,edgeb:4,edges1:1,edges2:1,edges3:1,edges_def:4,effect:[],eg:[],ego2tensor:[5,8],ego:[2,3,5,6],egograph:[2,6,8],either:[],element:[0,2,4,5,6],elimin:[],els:0,emb:5,embed:5,embedding_dim:5,embedding_encod:5,embeddingencod:[4,7],employe:0,empti:2,enabl:[],encod:5,encount:[],end:5,endpoint:[],enough:0,ensur:[],enter:[],entir:[],entri:1,environ:[],ep:[],epoch:[4,5,7],epsilon:5,equal:[],equival:4,error:[],etc:[],etyp:[],evalu:[],evaluate_gener:[],even:[],everi:5,everyth:[],exactli:[],examin:[],exampl:[0,1,4,5,7],example_tupl:[],except:[0,4],execut:7,exhaust:[],exist:2,exp:5,expand:6,expand_factor:[4,5,6],expect:[4,5],experiment:5,experimental_autocast:[],explain:[4,7,8],explain_result:[],explan:[4,5],expon:[],extern:[],fact:[],factor:6,fail:7,fall:[],fals:[0,2,4,5,6],faster:[],feat_concat:[4,7],featur:5,feed:[2,5],fetch:[0,2,7],file:4,filepath:[],find:[],fine:[],finish:[],first:5,fit:5,fit_gener:[],fix:[],flag:[],flat:[],flatten:[],float16:[],float32:[],fly:[],fn:[],follow:[0,6],form:[],format:[0,1,2,4,7],forward:[],found:[0,5,6],frac:5,fraction:[],framework:5,free:[],freq_th:[0,5,6],frequenc:[0,5,6],frequent:[0,5,6],fresh:[],from:[0,1,2,4,5,6,7],from_config:0,from_gener:5,from_logit:[4,7],full:[],fulli:[],g:[0,2,5,7],gate:5,gcn1:[4,7],gcn2:[4,7],gcn:5,gcn_model:4,gcnconv:[4,7],gener:[0,2,4,5],gensim:[],get:[0,1,3,5],get_config:[],get_dst_typ:0,get_edge_attr_data:0,get_edge_attr_info:0,get_edge_id:0,get_edge_prob:0,get_input_at:[],get_input_mask_at:[],get_input_shape_at:[],get_latest_training_loss:[],get_lay:[],get_losses_for:[],get_node_attr_data:0,get_node_attr_info:0,get_node_embed:5,get_node_label:0,get_output_at:[],get_output_mask_at:[],get_output_shape_at:[],get_src_typ:0,get_updates_for:[],get_weight:[],getter:[],give:[],given:[0,5],gnn:5,gnn_dev:4,gpadmin:4,gradient:[],gradienttap:[],gram:[2,5],graph1:4,graph2tensor:[0,1,2,3,4,5,6],graph:[2,3,5,6,7,8],graph_nam:[4,7],greater:[4,7],greatli:[],greedili:[],greenplum:4,ground:[],group:[],guarante:[],guid:[],h5:[],h5py:[],h:5,h_:5,h_i:5,h_j:5,h_k:5,h_u:5,h_v:5,ha:4,had:4,handl:5,hasattr:[],hash:[],hashfxn:[],hat:5,have:[3,5,6],hdf5:[],head:5,header:5,henc:[],here:[],heterogen:4,heterograph:4,hidden:5,hierarch:[],high:[],higher:[],highest:0,histori:4,home:[],hop:[2,5,6],horizont:[],host:[4,7],how:[2,4,5],hs:[],http:7,i:[0,5,6],id:[0,1,4,5,6,7],ident:[],identif:7,ie:[],ignor:[0,7],ij:5,immedi:[],implement:5,importerror:[],improv:[],inbound:[],inbound_nod:[],includ:5,include_edg:[2,4,5],include_optim:[],incom:[],incompat:[],increas:[],increment:7,indent:0,index:[4,5,7,8],indic:[1,2,5],individu:[],infer:[],infinit:[],inform:[],infrequ:[],inherit:[],init_ep:5,initi:[1,5,6],initial_epoch:[],input:5,input_dim:[4,5,7],input_mask:[],input_shap:[],input_signatur:[],input_spec:[],inputcontext:[],inputspec:[],insert:[],insid:[],instanc:[0,4,5],instanti:[],instead:[],instruct:[],int32:5,int64:5,int_feat:5,integ:5,integer_lookup:5,integerlookup:5,integerlookupencod:[4,7],intend:[],interact:[],interfac:4,intern:[],interpol:5,interpret:[],invalid:[],invers:[],io:[],is_edge_direct:0,is_head:0,is_node_label:0,isn:[],isomorph:5,issu:[],iter:[],its:[4,5,7],itself:[],iu:5,j:5,jitter:[],json:[0,7],json_str:[],k:5,k_:5,keep:[],keep_vocab_item:[],kei:[0,2,4,5],kera:[4,5],kernel:[],kernel_initi:[],keyword:[],kinmathc:[],know:[],kwarg:[0,2,4,5,6],l:5,label:[0,2,4],lambda:[],larg:[],larger:[],last:5,later:[],launch:[],layer:[4,8],layer_a:[],layer_b:[],layer_nam:[],layernorm:5,lead:1,leaky_relu:5,leakyrelu:5,learn:[],least:1,leav:[],left:[],len:[],length:[0,2,3,5,6],lesaint:[],leteli:[],level:8,like:6,limit:[],limits_:5,line:[],line_length:[],linear:5,linearli:[],linesent:[],list:[0,3,4,5,6,7],liusx:[],load:[],load_model:[],load_weight:[],local:[],locat:[],log:[],logic:[],longer:[],look:[],lookup:[0,2,5],lookup_edg:0,lookup_nod:0,loop:[4,5,7],loss:[4,7],loss_weight:[],low:[],lower:[],lpha:[],lpha_:[],m:[],machin:[],mae:[],mai:[],make:[],make_predict_funct:[],make_test_funct:[],make_train_funct:[],mani:[2,5],manipul:0,manual:[],map:[],mask:[],match:[5,6],math:[],math_op:[],mathbf:5,mathcal:5,mathemat:[],mathop:5,matmul:[],matrix:[],max:5,max_final_vocab:[],max_queue_s:[],max_vocab_s:[],maximum:[],mean:[4,5,7],meow:[],mere:[],messag:[5,7],message_pass:4,messagepass:4,meta:[3,6],meta_path:[4,5,6],metapath2vec:[],metapathrandomwalk:8,metapathsampl:[5,8],method:[],metric:[4,7],metric_1:[],metric_2:[],metrics_nam:[],might:[],million:[],min:5,min_alpha:[],min_count:[],mini:[0,1,3],minim:[],minimum:[],mismatch:[],mix:[],mixed_precis:[],mlp:5,mlp_activ:5,mlp_unit:5,mod:[],mode:5,model:[7,8],model_12345678:4,model_def:7,model_desc:7,model_from_json:[],model_from_yaml:[],model_id:[4,7],model_nam:7,model_save_dir:[],model_save_path:4,model_statu:7,modifi:[],modul:8,more:[0,4,5,6],most:5,most_similar:5,move:[],mse:[],multi:5,multi_heads_reduct:5,multicor:[],multipl:5,multiprocess:[],must:[0,5],my_metric_lay:[],my_model:[],my_modul:[],mylay:[],mymetriclay:[],mymodul:[],n1:6,n2:6,n3:6,n:[0,2,5],n_step:5,name:[0,2,4,5,7],name_scop:[],namedtupl:[],narrai:0,ndarrai:[0,1,2,5,6],ndim:[],need:[],neg:[2,5],negative_sampl:[2,5],negative_slop:5,neighbor:[0,6],neighbour:[0,1,5,6],nest:[],network:5,never:[],next:[],nice:[],node2vec:[0,6],node2vec_walk:0,node2vecwalk:[5,8],node:[0,2,3,4,5,6,7,8],node_attr:0,node_id:[],node_import:[],node_index:5,node_indic:5,node_label:[0,4],node_typ:[0,1],nodea:[1,4],nodeb:4,nodes_def:4,nois:[],non:[],non_trainable_vari:[],non_trainable_weight:[],none:[0,1,2,4,5,7],nor:[],normal:5,notabl:[],note:[],now:[],np:[0,1,2,5,6],ns_expon:[],null_word:[],num:[0,2],num_basis_block:5,num_employe:0,num_head:5,num_n:5,num_rel:5,num_target:2,num_thread:0,num_token:5,number:[0,2,5,6],numer:[],numpi:[0,1,2,6],numpy_graph:4,numpygraph:[4,6,8],object:[0,2,3,4,5],obtain:[],offset:[1,2,5],often:[],on_batch_begin:[],on_batch_end:[],on_read:[],on_test_batch_end:[],on_train_batch_end:[],one:[0,1,2,5],onehot:5,onehot_encod:5,ones:[],onli:[0,2,4,5,7],oov_token:5,op:[],oper:5,optim:[4,7],option:[0,1,4],order:[],org:[],origin:[],os:[],other:[0,4,5],other_tupl:[],otherwis:[0,2],out1:[4,7],out:[0,4,5,6],out_1_acc:[],out_1_loss:[],out_1_ma:[],out_acc:[],out_lay:5,out_layers_def:[4,7],out_loss:[],out_ma:[],outbound_nod:[],output:[4,5],output_1:[],output_2:[],output_a:[],output_b:[],output_dim:[4,5,7],output_mask:[],output_shap:[],output_signatur:5,over:5,overhead:[],overrid:[],overridden:[],overrightarrow:5,overwrit:[],own:4,p:[0,5,6],page:8,pai:[],pair:[0,1],paper:[4,5],paradigm:5,param:0,paramet:[0,1,2,3,4,5,6,7],parameterserverstrategi:[],part:[],particularli:[],pass:[4,5,6],password:4,path:[0,2,3,4,5,6],pathgener:[],pathlik:[],pattern:6,pdf:[],per:[],perform:6,phase:[],pick:[],picklabl:[],plan:[],pleas:[],point:7,polici:[],pool:5,popular:[],port:[4,7],posit:[],postgr:4,potenti:[],pre:[4,5],pre_proc_lay:5,pre_proc_layers_def:[4,7],preced:[],precis:[],predict:[5,8],predict_and_explain:8,predict_data:7,predict_gener:[],predict_on_batch:[],predict_step:[],prefer:[],prefix:[],preprocess:5,prevent:5,previou:[],prime:5,print:[0,1],print_fn:[],prob_column:4,prob_th:[4,7],probabl:[0,4,5,6,7],process:[4,5],produc:[],product:[],progbarlogg:[],program:5,progress:[],prompt:[],propag:[],properli:[],properti:[0,1,3],proport:[],provid:[],prune:[],put:[2,4],python:[],pythonhashse:[],q:[0,5,6],q_:5,queue:[],r:5,r_:5,rac:[],rais:0,ram:[],randint:[1,5],random:[0,1,4,5,6],random_walk:0,randomli:0,randomwalk:8,rang:[4,5,7],rank:[],rate:[],rather:[],rb:5,reach:[],real:[],reason:[],receiv:[4,5],recent:[],recommend:[],record:[],recurs:[],reduc:[4,5],reduce_lay:5,reduce_layer_def:[4,7],reduce_mean:[],reduce_sum:[],reduct:[4,5,7],refer:[],referenc:[],reflect:[],regular:5,reinstanti:[],rel:[],relat:[0,5,8],relation_target:0,relev:[],reli:[],relu:[4,5,7],remain:[],remedi:[],remov:[],repeat:5,replac:0,replica:[],repres:[0,2,3,5,6],represent:2,reproduc:[],requir:[0,5],reset:[],reset_metr:[],residu:5,resourcevari:[],respect:[1,4],restor:[],result:[4,5,7],resum:[],retriev:[],return_dict:[],reus:[],revers:0,right:7,rmsprop:[],root:[],round:[],routin:[],row:2,royo:[],rpc:0,rule:[],rule_default:[],rule_discard:[],rule_keep:[],run:[],run_eagerli:[],runtimeerror:[],s:[0,5,6],safe:[],sai:[],salari:0,same:[0,1,2,3,6],sampl:[0,2,5,6],sample_neighbor:[0,6],sample_weight:[],sampler:[5,8],sampler_process_num:[4,5],sampling_t:[2,5],save:4,save_format:[],save_model:[],save_trac:[],save_weight:[],saved_model:[],savedmodel:[],saveopt:[],scalar:[],scale:[],schedul:[],schema:[0,4,6],scope:[],search:8,second:[],section:[],see:[0,4,5,6,7],seed:[],seed_id:[],seed_label:[],select:[],self:5,sens:[],sentenc:[],separ:[],sequenc:[0,1,3,5,6],sequence_length:[],serial:0,serializ:[],serialization_and_sav:[],serv:8,server:0,set:2,set_weight:[],settabl:[],sever:[0,5],sg:[],shape:[2,5],shard:[],share:[],should:[0,1,3,5,6],shown:[],shuffl:4,sigma:5,sigmoid:5,signatur:[],silent:[],similar:5,simpli:[],sinc:[0,5],singl:[],situat:0,size:[0,1,2,5],skip:[2,5],skip_mismatch:[],skipgram:8,slightli:[],slope:5,slower:[],small:[],smoth:5,so:[],softmax:[],some:[],someth:6,soon:[],sort:[],sort_vocab:[],sorted_vocab:[],sourc:[0,1],spars:[],sparsecategoricalaccuraci:[4,7],sparsecategoricalcrossentropi:[4,7],special:[],specif:[],specifi:[0,1,5,6,7],spin:[],sqrt:5,src:[0,1,2,5,6],src_id:[0,1,2],src_type:[0,4],stabil:[],stage:[],standalon:[],standard:[],start:[0,6],start_from:7,state:[],state_upd:[],statefulli:[],statu:[],step:0,steps_per_epoch:[],steps_per_execut:[],still:[],stop:[],store:[],str:[0,1,2,4,5,6],str_feat:5,strategi:[0,4,5,6],stream:[],string:[5,6,7],string_lookup:5,stringlookup:5,structur:[],sub:[2,3,5],subclass:4,submodul:[],substitut:[],succe:7,success:[],suffix:[],suggest:[],sum:[4,5],sum_:5,summari:4,suppli:[],support:0,supports_mask:[],suppos:[4,5],symbol:[],synchron:[],t:5,tabl:[],take:[],taken:4,target:[0,2,4,5,7],target_class:[4,5,7],task:7,task_id:7,task_statu:7,tell:[],tempor:[],tensor:[2,5],tensorboard:4,tensorboard_log:4,tensorboard_log_dir:4,tensorboard_update_freq:[4,7],tensorflow:[2,5],tensorshap:[],tensorspec:5,termin:[],test:7,test_data:7,test_on_batch:[],test_step:[],text8corpu:[],text:[],tf:[4,5],than:[4,7],thei:[],them:[],themselv:[],thereof:[],thi:[1,5],third:[],thread:0,three:[],threshold:[0,4,5,6,7],through:[],thu:[1,2],time:7,timestep:[],tk:5,to_config:0,to_json:[],to_yaml:[],togeth:[],token:[],top:8,topk:0,topn:5,topolog:[],total:[2,5],tpu:[],trace:[],track:7,train:8,train_data:7,train_loop_def:[4,7],train_on_batch:[],train_step:[],trainabl:[],trainable_vari:[],trainable_weight:[],transfer:4,transform:5,travers:[],treat:[],trim:[],trim_rul:[],triplet:2,truncat:[],truth:[],tune:[],tupl:[0,2,3,4,5,6],tutori:[],two:[],type:[0,1,5,6],typeerror:[],typic:[],u:5,unambigu:[],unclear:[],under:[],understood:[],undirect:0,unexpect:[],unifi:5,uniform:[0,5,6],uniformli:0,uniniti:[],union:[1,2,6],uniqu:[],unit:[4,5,7],unknown:[],unless:[],unlik:[],unpack:[],unrel:[],unspecifi:[],unsupport:[],until:[],unus:[],up:[],updat:[],url:7,us:[0,4,5,6,7],use_bia:5,use_edge_prob:[0,5,6],use_multiprocess:[],use_resourc:[],user:4,user_id:7,usual:5,util:[],uv:5,v:5,v_:5,v_b:5,val_sample_weight:[],valid:[0,5,6],valid_s:4,validation_batch_s:[],validation_data:[],validation_freq:[],validation_split:[],validation_step:[],valu:[0,2,4,5],valueerror:0,variabl:[],variable_dtyp:[],variableaggreg:[],variablesynchron:[],varianc:[],vector:5,verbos:[],versa:[],via:[],vice:[],vid:[],vocab:[],vocab_s:5,vocabulari:[4,5,7],vocabulary_s:[2,5],vtype:[],w:5,w_:5,wa:7,wai:0,walk:[0,5,6],walk_length:[0,5,6],walker:6,want:[],we:[],weight:[0,2,5],weight_nam:[],weighted_metr:[],well:[],were:[],what:[],when:[0,5],whenev:[],where:5,whether:[0,2,4,5,6],which:[0,1,2,4,5,6,7],who:[],whose:[0,1,2,4,5,6],window:[2,5,7],window_s:[2,5],wish:[],with_name_scop:[],within:5,without:0,woof:[],word2vec:[],word:[],work:0,worker:[],workflow:[],workmat:0,wors:[],would:[4,7],wrap:[],writen:[],www:[],x0:[],x1:[],x:[],x_val:[],y:[],y_pred:[],y_true:[],y_val:[],yaml:[],yaml_str:[],year:[4,7],year_emb:[4,7],year_lookup:[4,7],yet:[],yield:[0,5],you:[],your:[],z:[],zero:[]},titles:["Graph","Nodes &amp; Edges","Converter","EgoGraph","Top Level API","Model","Sampler","Serving","Welcome to graph2tensor\u2019s documentation!"],titleterms:{"class":[0,1,2,3,5,6],api:[4,8],attrcompact:5,attrconcat:[],build_model:4,convert:2,creat:4,data:5,datagener:[],deepwalk:5,distnumpyattributestoreservic:0,distnumpygraph:0,distnumpyrelationstoreservic:0,document:8,edg:1,ego2tensor:2,egograph:3,egotensorgener:5,embeddingencod:5,evalu:4,explain:5,gatconv:5,gcnconv:5,ginconv:5,graph2tensor:8,graph:[0,4],graph_from_dgl:4,graph_from_pg:4,graph_to_dgl:4,indic:8,integerlookupencod:5,integratedgradi:5,layer:5,level:4,messagepass:5,metapath2vec:5,metapathrandomwalk:6,metapathsampl:6,model:[4,5],node2vec:5,node2vecwalk:6,node:1,numpygraph:0,onehotencod:5,pathgener:[],predict:[4,7],predict_and_explain:[4,7],randomwalk:6,refer:8,relat:4,rgcnconv:5,s:8,sampler:6,serv:7,skipgram:2,skipgramgener:[],skipgramgenerator4deepwalk:5,skipgramgenerator4metapath2vec:5,skipgramgenerator4node2vec:5,stringlookupencod:5,tabl:8,top:4,train:[4,7],unimp:5,welcom:8}})