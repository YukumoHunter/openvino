// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/op/einsum.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {
OutputVector translate_matrix_set_diag_op(const NodeContext& node) {
    default_op_checks(node, 1, {"MatrixSetDiagV3"});

    auto op_type = node.get_op_type();
    TENSORFLOW_OP_VALIDATION(node,
                             op_type == "MatrixSetDiagV3",
                             "Internal error: incorrect usage of translate_matrix_set_diag_op.");

    auto input = node.get_input(0);
    auto diagonal = node.get_input(1);
    auto align = node.get_attribute<string>("align");

    return {input};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
