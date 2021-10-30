// Copyright (C) 2020-2021 Intel Corporation
//
// SPDX-License-Identifier: MIT

import React, { useState } from 'react';
import { useDispatch } from 'react-redux';
import moment from 'moment';
import { Row, Col } from 'antd/lib/grid';
import Title from 'antd/lib/typography/Title';
import Text from 'antd/lib/typography/Text';

import getCore from 'cvat-core-wrapper';
import { Project } from 'reducers/interfaces';
import { updateProjectAsync } from 'actions/projects-actions';
import LabelsEditor from 'components/labels-editor/labels-editor';
import BugTrackerEditor from 'components/task-page/bug-tracker-editor';
import UserSelector from 'components/task-page/user-selector';

const core = getCore();

interface DetailsComponentProps {
    project: Project;
}

export default function DetailsComponent(props: DetailsComponentProps): JSX.Element {
    const { project } = props;

    const dispatch = useDispatch();
    const [projectName, setProjectName] = useState(project.name);

    return (
        <div style={{ marginLeft: '41%', fontSize: '40px' }}>
             {projectName}
        </div>
    );
}