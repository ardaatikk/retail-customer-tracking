from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import pytest
from src.analytics.input import load_selected_tracks
from src.analytics.summary import calculate_customer_summary, calculate_zone_summary
from src.pipeline import analytics_pipeline as pipeline


def data(camera='live1', capture='session1'):
    return pd.DataFrame([dict(camera_id=camera, capture_id=capture, customer_id=7,
                              timestamp=t, zone=None, ground_x=10, ground_y=20)
                         for t in [1., 3.]])


def test_custom_input_infers_scope_and_summary_identity(tmp_path):
    p=tmp_path/'tracks.parquet'
    data().to_parquet(p)
    selected=load_selected_tracks(p)
    summary=calculate_customer_summary(selected)
    assert summary.iloc[0].camera_id == 'live1'
    assert summary.iloc[0].capture_id == 'session1'
    assert summary.iloc[0].store_time_seconds == 2
    assert calculate_zone_summary(selected).empty


def test_multi_session_requires_selection(tmp_path):
    p=tmp_path/'tracks.parquet'
    pd.concat([data(),data(capture='session2')]).to_parquet(p)
    with pytest.raises(ValueError, match='exactly one'):
        load_selected_tracks(p)
    assert len(load_selected_tracks(p,capture_id='session2')) == 2
    with pytest.raises(ValueError, match='No observations'):
        load_selected_tracks(p,camera_id='unknown')


def test_reused_customer_ids_do_not_merge_across_sessions():
    first, second=data(), data(capture='session2')
    second['timestamp']=[100.,105.]
    first['zone']='Main'
    second['zone']='Main'
    mixed=pd.concat([first,second])
    summary=calculate_customer_summary(mixed)
    assert summary.store_time_seconds.tolist() == [2.,5.]
    zones=calculate_zone_summary(mixed)
    assert zones.total_dwell_seconds.tolist() == [2.,5.]
    assert zones.capture_id.tolist() == ['session1','session2']


def test_raw_final_ids_preserve_preconfirmation_time(tmp_path):
    frame=data()
    frame['customer_id']=pd.Series([None,7],dtype='Int64')
    frame['final_customer_id']=pd.Series([7,7],dtype='Int64')
    p=tmp_path/'raw.parquet'
    frame.to_parquet(p)
    selected=load_selected_tracks(p)
    assert selected.customer_id.tolist() == [7,7]
    assert calculate_customer_summary(selected).iloc[0].store_time_seconds == 2


def test_no_confirmed_rows_rejected(tmp_path):
    frame=data()
    frame['customer_id']=pd.Series([None,None],dtype='Int64')
    p=tmp_path/'tracks.parquet'
    frame.to_parquet(p)
    with pytest.raises(ValueError, match='No confirmed'):
        load_selected_tracks(p)


def test_custom_no_reference_skips_spatial_and_zone_outputs(tmp_path, capsys):
    source=tmp_path/'tracks.parquet'
    data().to_parquet(source)
    pipeline.run_analytics_pipeline(tracks_path=source)
    destination=tmp_path/'analytics'
    assert (destination/'customer_summary.parquet').exists()
    assert not (destination/'zone_summary.parquet').exists()
    assert not (destination/'trajectory.png').exists()
    assert 'Spatial plots skipped' in capsys.readouterr().out


def test_reference_enables_zone_free_spatial_plots(tmp_path):
    source=tmp_path/'tracks.parquet'
    data().to_parquet(source)
    image=tmp_path/'reference.png'
    cv2.imwrite(str(image),np.zeros((50,50,3),dtype=np.uint8))
    destination=tmp_path/'output'
    pipeline.run_analytics_pipeline(tracks_path=source,reference_frame=image,output_dir=destination)
    assert (destination/'trajectory.png').exists()
    assert (destination/'heatmap.png').exists()
    assert cv2.imread(str(destination/'trajectory.png')).shape[:2] == (50,50)
    assert not (destination/'zone_summary.parquet').exists()


def test_default_showroom_route_preserved(tmp_path,monkeypatch):
    import src.analytics.summary as summary
    source=tmp_path/'tracks.parquet'
    frame=data(camera='Showroom',capture='capture1')
    frame['zone']='Main'
    frame.to_parquet(source)
    monkeypatch.setattr(summary,'TRACKS_PATH',source)
    plots=[]
    monkeypatch.setattr(pipeline,'generate_trajectory',lambda **kw:plots.append(kw))
    monkeypatch.setattr(pipeline,'generate_heatmap',lambda **kw:plots.append(kw))
    output=tmp_path/'output'
    pipeline.run_analytics_pipeline(output_dir=output)
    assert len(plots)==2 and plots[0]['camera_id']=='Showroom'
    assert (output/'zone_summary.parquet').exists()
