// Copyright (C) 2020-2021 Intel Corporation
//
// SPDX-License-Identifier: MIT

import React from 'react';
import { RouteComponentProps } from 'react-router';
import { withRouter } from 'react-router-dom';
import Text from 'antd/lib/typography/Text';
import { Row, Col } from 'antd/lib/grid';
import Button from 'antd/lib/button';
import Icon from '@ant-design/icons';
import Dropdown from 'antd/lib/dropdown';
import Progress from 'antd/lib/progress';
import moment from 'moment';
import Title from 'antd/lib/typography/Title';

import ActionsMenuContainer from 'containers/actions-menu/actions-menu';
import { ActiveInference } from 'reducers/interfaces';
import { MenuIcon } from 'icons';
import AutomaticAnnotationProgress from './automatic-annotation-progress';
import { size } from 'lodash';

export interface TaskItemProps {
    taskInstance: any;
    previewImage: string;
    deleted: boolean;
    hidden: boolean;
    activeInference: ActiveInference | null;
    cancelAutoAnnotation(): void;
}

class TaskItemComponent extends React.PureComponent<TaskItemProps & RouteComponentProps> {
    private renderPreview(): JSX.Element {
        const { previewImage } = this.props;
        return (
            <Col span={4}>
                <div className='cvat-task-item-preview-wrapper'>
                    <img alt='Preview' className='cvat-task-item-preview' src={previewImage} />
                </div>
            </Col>
        );
    }

    private renderDescription(): JSX.Element {
        // Task info
        const { taskInstance } = this.props;
        const { id } = taskInstance;
        const description = taskInstance.subset;
        const owner = taskInstance.owner ? taskInstance.owner.username : null;
        const updated = moment(taskInstance.updatedDate).fromNow();
        const created = moment(taskInstance.createdDate).format('MMMM Do YYYY');

        // Get and truncate a task name
        const name = `${taskInstance.name.substring(0, 70)}${taskInstance.name.length > 70 ? '...' : ''}`;
        return (
            <Col span={14}>
                <div style={{ fontSize: '20px', fontStyle: 'unset' }}>{name}</div>
            </Col>
        );
    }

    private renderNavigation(): JSX.Element {
        const { taskInstance, history } = this.props;
        const { jobs, id: taskId } = taskInstance;

        return (
            <Col span={6}>
                <Button
                    style={{ width: '150px', marginTop: '10%' }}
                    //type='ghost'
                    size='large'
                    href={`/tasks/${taskId}/jobs/${jobs[0].id}`}
                    onClick={(e: React.MouseEvent): void => {
                        e.preventDefault();
                        history.push(`/tasks/${taskId}/jobs/${jobs[0].id}`);
                    }}
                >
                    Start
                </Button>

            </Col>
        );
    }

    public render(): JSX.Element {
        const { deleted, hidden } = this.props;
        const { taskInstance, history } = this.props;
        const { jobs, id: taskId } = taskInstance;
        const name = `${taskInstance.name.substring(0, 70)}${taskInstance.name.length > 70 ? '...' : ''}`;
        const style = {};
        if (deleted) {
            (style as any).pointerEvents = 'none';
            (style as any).opacity = 0.5;
        }

        if (hidden) {
            (style as any).display = 'none';
        }

        return (
            <Button
                    style={{ width: '150px', height: '150px', marginRight: '5%' }}
                    size='large'
                    href={`/tasks/${taskId}/jobs/${jobs[0].id}`}
                    onClick={(e: React.MouseEvent): void => {
                        e.preventDefault();
                        history.push(`/tasks/${taskId}/jobs/${jobs[0].id}`);
                    }}
            >
                    <div style={{ marginTop: '45%' }}>{name}</div>
            </Button>
            //<Row className='cvat-tasks-list-item' justify='center' align='top' style={{ ...style }}>
            //    {this.renderDescription()}
            //    {this.renderNavigation()}
           // </Row>
        );
    }
}

export default withRouter(TaskItemComponent);
