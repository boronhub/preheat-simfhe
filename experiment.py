from math import ceil, exp, log
from dataclasses import dataclass, field
from typing import List, Dict
import csv
import sys
import tqdm
import tabulate
import copy

import profiler
import params
import post_process

RPT_ATTR = {
    "total ops": "sw.total_ops",
    "total mult": "sw.mult",
    "total add": "sw.add",
    "total ntt": "sw.ntt",
    "dram total": "arch.dram_total_rdwr_small",
    "dram limb rd": "arch.dram_limb_rd",
    "dram limb wr": "arch.dram_limb_wr",
    "dram key rd": "arch.dram_auto_rd",
    "total cycles (slow, worst case)": "arch.total_cycle_sm_wc",
    "total cycles (slow, best case)": "arch.total_cycle_sm_bc",
    "total cycles (fast, worst case)": "arch.total_cycle_fm_wc",
    "total cycles (fast, best case)": "arch.total_cycle_fm_bc",
}


@dataclass
class Target:
    name: str
    depth: int
    args: List = field(default_factory=list)
    kwargs: List = field(default_factory=dict)


global_profiler = None

def generate_profile(target: Target):
    global global_profiler
    if global_profiler is None:
        global_profiler = profiler.Profiler(target.name)
    global_profiler.profile(target.name, *target.args, **target.kwargs)
    return global_profiler


def generate_flamegraph(experiment: profiler.Profiler, attr, suffix=""):
    graph_name = experiment.name + f"_{attr}"
    if suffix:
        graph_name += f"_{suffix}"
    post_process.flamegraph(graph_name, experiment.data, attr)


def get_table(data, attr_dict, depth):
    table = post_process.get_table(data, attr_dict.values(), depth)
    # transpose
    ttable = []
    nrow, ncol = len(table[0]), len(table)
    for row_idx in range(nrow):
        ttable.append([table[col_idx][row_idx] for col_idx in range(ncol)])
    return ttable


def save_csv(headers, data, filepath):
    #headers = ["logN", "dnum", "fft_iters", "fresh_limbs", "op_count", "total_mem"]
    with open(filepath, "w") as csvfile:
        csvwriter = csv.writer(csvfile, dialect="excel")
        csvwriter.writerow(headers)
        csvwriter.writerows(data)


def get_headers(attr_dict):
    return ["fn"] + list(attr_dict.keys())


def run_single(target, attr_dict=RPT_ATTR):
    experiment = generate_profile(target)
    acc_data = post_process.accumulate(experiment.data)
    data = get_table(acc_data, attr_dict, target.depth)
    headers = get_headers(attr_dict)
    return (headers, data)


def run_multiple(targets, attr_dict=RPT_ATTR, gen_graph=False):
    cum_data = []
    headers = get_headers(attr_dict)
    for target in tqdm.tqdm(targets):
        experiment = generate_profile(target)
        if gen_graph:
            for attr in attr_dict.values():
                generate_flamegraph(experiment, attr)
        acc_data = post_process.accumulate(experiment.data)
        data = get_table(acc_data, attr_dict, target.depth)
        cum_data += data
    return (headers, cum_data)


def compare_bootstrap(schemes, attr_dict=RPT_ATTR):
    cum_data = []
    headers = get_headers(attr_dict)
    for scheme_params in schemes:
        target = Target("bootstrap.bootstrap", 1, [scheme_params])
        experiment = generate_profile(target)
        acc_data = post_process.accumulate(experiment.data)
        data = get_table(acc_data, attr_dict, target.depth)
        cum_data += data
    return (headers, cum_data)


def print_table(headers, data, print_csv=False):
    if print_csv:
        writer = csv.writer(sys.stdout)
        writer.writerow(headers)
        writer.writerows(data)
    else:
        tabulate.PRESERVE_WHITESPACE = True
        print(tabulate.tabulate(data, headers=headers))
        tabulate.PRESERVE_WHITESPACE = False


def aux_subroutine_benchmarks(scheme_params: params.SchemeParams):
    micro_args = [scheme_params.mod_raise_ctxt, scheme_params]
    targets = [
        Target("micro_benchmarks.mod_up", 1, micro_args),
        Target("micro_benchmarks.mod_down", 1, micro_args),
        Target("micro_benchmarks.decomp", 1, micro_args),
        Target("micro_benchmarks.inner_product", 1, micro_args),
        Target("micro_benchmarks.automorph", 1, micro_args),
    ]
    headers, data = run_mutiple(targets)
    print_table(headers, data)
    save_csv(headers, data, "data/aux_subroutine.csv")


def low_level_benchmark(scheme_params: params.SchemeParams):
    micro_args = [scheme_params.mod_raise_ctxt, scheme_params]
    targets = [
        Target("micro_benchmarks.pt_add", 1, micro_args),
        Target("micro_benchmarks.add", 1, micro_args),
        Target("micro_benchmarks.pt_mult", 1, micro_args),
        Target("micro_benchmarks.mult", 1, micro_args),
        Target("micro_benchmarks.rotate", 1, micro_args),
        Target("micro_benchmarks.hoisted_rotate", 1, micro_args),
    ]
    headers, data = run_mutiple(targets)
    print_table(headers, data)
    save_csv(headers, data, "data/low_level.csv")


def high_level_benchmark(scheme_params: params.SchemeParams):
    targets = [
        Target("fft.fft", 2, [scheme_params.mod_raise_ctxt, scheme_params]),
        Target("eval_sine.eval_sine", 2, [scheme_params.cts_ctxt, scheme_params]),
    ]
    headers, data = run_mutiple(targets)
    print_table(headers, data)
    save_csv(headers, data, "data/high_level.csv")


def bootstrap_benchmark(scheme_params: params.SchemeParams, rpt_depth=3):
    targets = [Target("bootstrap.bootstrap", rpt_depth, [scheme_params])]
    headers, data = run_mutiple(targets)
    print_table(headers, data)
    save_csv(headers, data, "data/bootstrap.csv")


def fft_best_params():
    """
    Sweep for each logN, 16 and 17
    for each dnum from 1 to 6
    for each squashing 1 to 5
    """

    logNVals = [16, 17]
    dnum_vals = range(1, 7)
    squashing_vals = range(1, 7)
    total_runs = len(logNVals) * len(dnum_vals) * len(squashing_vals)

    table = []
    with tqdm.tqdm(total=total_runs) as pbar:
        for logN in logNVals:
            for dnum in dnum_vals:
                fft_iter_vals = [int(ceil((logN - 1) / x)) for x in squashing_vals]
                for fft_iters in fft_iter_vals:
                    scheme_params = params.SchemeParams(
                        logN=logN,
                        dnum=dnum,
                        fft_iters=fft_iters,
                        fft_style=params.FFTStyle.UNROLLED_HOISTED,
                        arch_param=params.BEST_ARCH_PARAMS,
                    )

                    try:
                        start_limbs = scheme_params.bootstrapping_Q0_limbs
                    except ValueError:
                        pbar.update(1)
                        continue

                    # target = Target(
                    #     "bootstrap.fft", [scheme_params.mod_raise_ctxt, scheme_params]
                    # )
                    target = Target("bootstrap.bootstrap", 1, [scheme_params])
                    experiment = generate_profile(target)
                    acc_data = post_process.accumulate(experiment.data)

                    op_count = post_process.get_attr(acc_data, "sw.total_ops", 1)
                    total_mem = post_process.get_attr(
                        acc_data, "arch.dram_total_rdwr_small", 1
                    )

                    table.append(
                        [
                            logN,
                            dnum,
                            fft_iters,
                            scheme_params.fresh_limbs,
                            op_count,
                            total_mem,
                        ]
                    )

                    pbar.update(1)

    headers = ["logN", "dnum", "fft_iters", "fresh_limbs", "op_count", "total_mem"]

    print_table(headers, table)
    save_csv(headers, table, "data/fft.csv")


def sweep_params_for_preheat():
    attributes = {
        "total ops": "sw.total_ops",
        "total mult": "sw.mult",
        "dram total": "arch.dram_total_rdwr_small",
        "dram limb rd": "arch.dram_limb_rd",
        "dram limb wr": "arch.dram_limb_wr",
        "dram key rd": "arch.dram_auto_rd",
        "total cycles (slow, worst case)": "arch.total_cycle_sm_wc",
        "total cycles (slow, best case)": "arch.total_cycle_sm_bc",
        # "total cycles (fast, worst case)": "arch.total_cycle_fm_wc",
        # "total cycles (fast, best case)": "arch.total_cycle_fm_bc",
    }

    base_arch_params = params.ArchParam(
        karatsuba=True,
        key_compression=True,
        rescale_fusion=True,
        cache_style=params.CacheStyle.ALPHA,
        mod_down_reorder=True,
    )

    scheme_params = []
    extra_columns = []
    extra_column_names = [
        "funits",
        "logq",
        "dnum",
        "cache_size",
    ]
    for logq in range(45, 51):
        for dnum in range(2, 7):
            for funits in range(10, 16):
                arch_param = copy.copy(base_arch_params)
                arch_param.funits = funits
                scheme_param = params.SchemeParams(
                    logq=logq,
                    logN=17,
                    dnum=dnum,
                    fft_iters=6,
                    fft_style=params.FFTStyle.UNROLLED_HOISTED,
                    arch_param=arch_param,
                )
                cache_size = scheme_param.get_max_cache_size()
                extra_columns.append(
                    [
                        funits,
                        logq,
                        dnum,
                        cache_size,
                    ]
                )
                scheme_params.append(scheme_param)

    targets = []
    for scheme_param in scheme_params:
        targets.append(Target("bootstrap.bootstrap", 1, [scheme_param]))

    headers, data = run_multiple(targets, attr_dict=attributes, gen_graph=False)

    for i in range(len(data)):
        data[i] += extra_columns[i]
    headers += extra_column_names

    print_table(headers, data, print_csv=True)


if __name__ == "__main__":
    sweep_params_for_preheat()
