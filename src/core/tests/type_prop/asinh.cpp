// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "unary_ops.hpp"

using Type = ::testing::Types<ov::op::v3::Asinh>;

INSTANTIATE_TYPED_TEST_SUITE_P(type_prop_asinh, UnaryOperator, Type);
