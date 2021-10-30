// Copyright (C) 2020-2021 Intel Corporation
//
// SPDX-License-Identifier: MIT

import React from 'react';
import { Col } from 'antd/lib/grid';
import Select from 'antd/lib/select';
import Button from 'antd/lib/button';

import { useSelector } from 'react-redux';

import {
    CombinedState,
} from 'reducers/interfaces';

interface Props {
    showFilters(): void;
}

function RightGroup(props: Props): JSX.Element {
    const {
        showFilters,
    } = props;

    const filters = useSelector((state: CombinedState) => state.annotation.annotations.filters);

    return (
        <Col>
        </Col>
    );
}

export default React.memo(RightGroup);
