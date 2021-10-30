// Copyright (C) 2020-2021 Intel Corporation
//
// SPDX-License-Identifier: MIT

import React from 'react';
import { Col } from 'antd/lib/grid';
import Icon from '@ant-design/icons';
import Popover from 'antd/lib/popover';
import Button from 'antd/lib/button';

import CVATTooltip from 'components/common/cvat-tooltip';

import {
    FirstIcon,
    BackJumpIcon,
    PreviousIcon,
    PreviousFilteredIcon,
    PreviousEmptyIcon,
    PlayIcon,
    PauseIcon,
    NextIcon,
    NextFilteredIcon,
    NextEmptyIcon,
    ForwardJumpIcon,
    LastIcon,
} from 'icons';

interface Props {
    playing: boolean;
    playPauseShortcut: string;
    nextFrameShortcut: string;
    previousFrameShortcut: string;
    forwardShortcut: string;
    backwardShortcut: string;
    prevButtonType: string;
    nextButtonType: string;
    onSwitchPlay(): void;
    onPrevFrame(): void;
    onNextFrame(): void;
    onForward(): void;
    onBackward(): void;
    onFirstFrame(): void;
    onLastFrame(): void;
    setPrevButton(type: 'filtered'): void;
    setNextButton(type: 'filtered'): void;
}

function PlayerButtons(props: Props): JSX.Element {
    const {
        playing,
        playPauseShortcut,
        nextFrameShortcut,
        previousFrameShortcut,
        forwardShortcut,
        backwardShortcut,
        prevButtonType,
        nextButtonType,
        onSwitchPlay,
        onPrevFrame,
        onNextFrame,
        onForward,
        onBackward,
        onFirstFrame,
        onLastFrame,
        setPrevButton,
        setNextButton,
    } = props;

    const prevRegularText = 'Go back';
    const prevFilteredText = 'Back';
    const prevEmptyText = 'Go back to an empty frame';
    const nextRegularText = 'Go next';
    const nextFilteredText = 'Forward';
    const nextEmptyText = 'Go next to an empty frame';

    let prevButton = <Button className='cvat-player-previous-button-filtered' onClick={onPrevFrame} />;
    let prevButtonTooltipMessage = prevFilteredText;
    setPrevButton('filtered');
    if (prevButtonType === 'filtered') {
        prevButton = (
            <Button style={{ width: '120px' }} type='primary' className='cvat-player-previous-button-filtered' onClick={onPrevFrame}>Previous case</Button>
        );
        prevButtonTooltipMessage = prevFilteredText;
    }

    let nextButton = <Button className='cvat-player-next-button-filtered' onClick={onNextFrame} />;
    let nextButtonTooltipMessage = nextFilteredText;
    setNextButton('filtered');
    if (nextButtonType === 'filtered') {
        nextButton = (
            <Button style={{ width: '120px' }} type='primary' className='cvat-player-next-button-filtered' onClick={onNextFrame}>Next case</Button>
        );
        nextButtonTooltipMessage = nextFilteredText;
    }

    return (
        <Col className='cvat-player-buttons'>
            <CVATTooltip placement='top' title={`${prevButtonTooltipMessage} ${previousFrameShortcut}`}>
                {prevButton}
            </CVATTooltip>
            <div style={{ width: '10px' }}> </div>
            <CVATTooltip placement='top' title={`${nextButtonTooltipMessage} ${nextFrameShortcut}`}>
                {nextButton}
            </CVATTooltip>
            <div style={{ width: '10px' }}> </div>
        </Col>
    );
}

export default React.memo(PlayerButtons);