// Copyright (C) 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT

import React from 'react';
import { useDispatch } from 'react-redux';
import Modal from 'antd/lib/modal';
import Menu from 'antd/lib/menu';
import fetch from 'node-fetch';
import { deleteProjectAsync } from 'actions/projects-actions';

interface Props {
    projectInstance: any;
}

export default function ProjectActionsMenuComponent(props: Props): JSX.Element {
    const { projectInstance } = props;

    const dispatch = useDispatch();

    const onDeleteProject = (): void => {
        Modal.confirm({
            title: `The project ${projectInstance.id} will be deleted`,
            content: 'All related data (images, annotations) will be lost. Continue?',
            className: 'cvat-modal-confirm-remove-project',
            onOk: () => {
                dispatch(deleteProjectAsync(projectInstance));
            },
            okButtonProps: {
                type: 'primary',
                danger: true,
            },
            okText: 'Delete',
        });
    };

    const onSyncProject = (): void => {
        Modal.confirm({
            title: `The project ${projectInstance.id} will be synchronized with the storage`,
            content: 'New images will be downloaded. Continue?',
            className: 'cvat-modal-confirm-sync-project',
            onOk: () => {
                const fetch = require('node-fetch');
                    const body = {project: projectInstance.name};
                    fetch('http://3.143.220.139:5000/transfer_data', {
                        method: "POST",
                        mode: 'cors',
                        body: JSON.stringify(body),
                        headers: {'Content-Type': "application/json"}
                    }).then(res => res.json()).then(json => console.log(json));
            },
            okButtonProps: {
                type: 'primary',
                danger: true,
            },
            okText: 'Synchronize',
        });
    };

    return (
        <Menu className='cvat-project-actions-menu'>
            <Menu.Item onClick={onSyncProject}>Synchronize</Menu.Item>
            <Menu.Item onClick={onDeleteProject}>Delete</Menu.Item>
        </Menu>
    );
}
