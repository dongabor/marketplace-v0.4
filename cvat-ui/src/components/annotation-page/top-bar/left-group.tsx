// Copyright (C) 2020-2021 Intel Corporation
//
// SPDX-License-Identifier: MIT

import React from 'react';
import { Col } from 'antd/lib/grid';
import Icon from '@ant-design/icons';
import Button from 'antd/lib/button';
import Dropdown from 'antd/lib/dropdown';

import AnnotationMenuContainer from 'containers/annotation-page/top-bar/annotation-menu';
import {
    MainMenuIcon,
} from 'icons';

import { useSelector } from 'react-redux';

import {
    CombinedState,
} from 'reducers/interfaces';

interface Props {
    showFilters(): void;
}

function LeftGroup(props: Props): JSX.Element {
    const {
        showFilters,
    } = props;

    const filters = useSelector((state: CombinedState) => state.annotation.annotations.filters);

    return (
        <Col className='cvat-annotation-header-left-group'>
            <Dropdown overlay={<AnnotationMenuContainer />}>
                <Button type='link' className='cvat-annotation-header-button'>
                    <Icon component={MainMenuIcon} />
                    Menu
                </Button>
            </Dropdown>
            <Button
                //type='secondary'
                style={{ marginTop: '3%', width: '225px', marginLeft: '10%' }}
                //className={`cvat-annotation-header-button ${filters.length ? 'filters-armed' : ''}`}
                onClick={showFilters}
            >
                Create filter
            </Button>
        </Col>
    );
}

export default React.memo(LeftGroup);
