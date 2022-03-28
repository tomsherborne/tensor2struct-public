{
    local exp_id = 0,
    project: "en_spider_mbert_dgreptile_norm",
    logdir: "log/%s/mbert_value_%d" % [self.project, exp_id],
    model_config: "configs/spider/model_config/en_spider_bert_value.jsonnet",
    model_config_args: {
        # data 
        use_other_train: true,

        # model
        num_layers: 6,
        sc_link: true,
        cv_link: true,
        loss_type: "softmax", # softmax, label_smooth

        # bert
        opt: "torchAdamw",   # bertAdamw, torchAdamw
        lr_scheduler: "bert_warmup_polynomial_group_v2", # bert_warmup_polynomial_group,bert_warmup_polynomial_grou_v2
        bert_token_type: true,
        bert_version: "bert-base-multilingual-uncased",
        bert_lr: 2e-5,

        # grammar
        include_literals: true,

        # training
        bs: 16,
        att: 0,
        lr: 5e-4,
        clip_grad: 1,
        num_batch_accumulated: 1,
        max_steps: 20000,
        save_threshold: 10000,
        use_bert_training: true,
        device: "cuda:0",
        do_cross_gradient_normalization: true, 

        # meta train
        meta_opt: "sgd",
        meta_lr: 1e-4,
        num_batch_per_train: 3, # This is the sum of total inner and outer if yield_outer_batch=True else this should be ==inner_steps
        inner_steps: 2,
        yield_outer_batch: true,
        data_scheduler: "db_reptile_scheduler",
        first_order: true
    },

    eval_section: "val",
    eval_type: "all", # match, exec, all
    eval_method: "spider_beam_search_with_heuristic",
    eval_output: "ie_dir/%s" % self.project,
    eval_beam_size: 3,
    eval_debug: false,
    eval_name: "%s_mbert_run_%d_k_%d_%s_%s_%d" % [self.project, exp_id, self.model_config_args.inner_steps,self.eval_section, self.eval_method, self.eval_beam_size],

    local _start_step = $.model_config_args.save_threshold / 1000,
    local _end_step = $.model_config_args.max_steps / 1000,
    eval_steps: [ 1000 * x for x in std.range(_start_step, _end_step)],
}