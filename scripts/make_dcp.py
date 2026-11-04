#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Wrap a directory of DCI J2C frames into a SMPTE picture MXF (asdcp-wrap) and a
minimal SMPTE DCP (CPL, PKL, ASSETMAP, VOLINDEX) so it can be checked with
libdcp's dcpverify or played on a cinema server.

usage: make_dcp.py FRAMES_DIR OUT_DCP_DIR [--fps 24] [--title NAME]
"""
import argparse
import base64
import datetime
import hashlib
import os
import subprocess
import sys
import uuid


def sha1_b64(path):
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return base64.b64encode(h.digest()).decode()


def j2c_size(path):
    with open(path, 'rb') as f:
        d = f.read(64)
    # SOC (2) SIZ marker (2) Lsiz (2) Rsiz (2) Xsiz (4) Ysiz (4)
    w = int.from_bytes(d[8:12], 'big')
    h = int.from_bytes(d[12:16], 'big')
    return w, h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('frames')
    ap.add_argument('out')
    ap.add_argument('--fps', type=int, default=24)
    ap.add_argument('--title', default='GAIL_TEST_FTR-1_F_51_2K_20261029_SMPTE_OV')
    ap.add_argument('--lenient', action='store_true', help='let asdcp-wrap accept frames with unequal parameters')
    a = ap.parse_args()

    frames = sorted(f for f in os.listdir(a.frames) if f.endswith('.j2c'))
    if not frames:
        sys.exit('no .j2c frames found')
    w, h = j2c_size(os.path.join(a.frames, frames[0]))
    os.makedirs(a.out, exist_ok=True)

    mxf_id = str(uuid.uuid4())
    mxf_name = f'j2c_{mxf_id}.mxf'
    mxf_path = os.path.join(a.out, mxf_name)
    if os.path.exists(mxf_path):
        os.remove(mxf_path)
    cmd = ['asdcp-wrap', '-L', '-p', str(a.fps), '-a', mxf_id]
    if a.lenient:
        cmd.append('-Z')
    subprocess.run(cmd + [a.frames, mxf_path], check=True)

    now = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()
    n = len(frames)
    cpl_id, pkl_id, am_id, reel_id = (str(uuid.uuid4()) for _ in range(4))
    cpl = f'''<?xml version="1.0" encoding="UTF-8"?>
<CompositionPlaylist xmlns="http://www.smpte-ra.org/schemas/429-7/2006/CPL">
  <Id>urn:uuid:{cpl_id}</Id>
  <AnnotationText>{a.title}</AnnotationText>
  <IssueDate>{now}</IssueDate>
  <Issuer>Gail</Issuer>
  <Creator>Gail GPU JPEG 2000 encoder</Creator>
  <ContentTitleText>{a.title}</ContentTitleText>
  <ContentKind>test</ContentKind>
  <ContentVersion>
    <Id>urn:uuid:{uuid.uuid4()}</Id>
    <LabelText>{a.title}</LabelText>
  </ContentVersion>
  <RatingList/>
  <ReelList>
    <Reel>
      <Id>urn:uuid:{reel_id}</Id>
      <AssetList>
        <MainMarkers>
          <Id>urn:uuid:{uuid.uuid4()}</Id>
          <EditRate>{a.fps} 1</EditRate>
          <IntrinsicDuration>{n}</IntrinsicDuration>
          <MarkerList>
            <Marker>
              <Label>FFOC</Label>
              <Offset>1</Offset>
            </Marker>
            <Marker>
              <Label>LFOC</Label>
              <Offset>{n - 1}</Offset>
            </Marker>
          </MarkerList>
        </MainMarkers>
        <MainPicture>
          <Id>urn:uuid:{mxf_id}</Id>
          <EditRate>{a.fps} 1</EditRate>
          <IntrinsicDuration>{n}</IntrinsicDuration>
          <EntryPoint>0</EntryPoint>
          <Duration>{n}</Duration>
          <Hash>{sha1_b64(mxf_path)}</Hash>
          <FrameRate>{a.fps} 1</FrameRate>
          <ScreenAspectRatio>{w} {h}</ScreenAspectRatio>
        </MainPicture>
        <meta:CompositionMetadataAsset xmlns:meta="http://www.smpte-ra.org/schemas/429-16/2014/CPL-Metadata">
          <Id>urn:uuid:{uuid.uuid4()}</Id>
          <EditRate>{a.fps} 1</EditRate>
          <IntrinsicDuration>{n}</IntrinsicDuration>
          <meta:FullContentTitleText>{a.title}</meta:FullContentTitleText>
          <meta:ReleaseTerritory>US</meta:ReleaseTerritory>
          <meta:VersionNumber status="final">1</meta:VersionNumber>
          <meta:Chain>Gail</meta:Chain>
          <meta:Distributor>Gail</meta:Distributor>
          <meta:Facility>Gail</meta:Facility>
          <meta:Luminance units="foot-lambert">14</meta:Luminance>
          <meta:MainSoundConfiguration>51/L,R,C,LFE,Ls,Rs</meta:MainSoundConfiguration>
          <meta:MainSoundSampleRate>48000 1</meta:MainSoundSampleRate>
          <meta:MainPictureStoredArea>
            <meta:Width>{w}</meta:Width>
            <meta:Height>{h}</meta:Height>
          </meta:MainPictureStoredArea>
          <meta:MainPictureActiveArea>
            <meta:Width>{w}</meta:Width>
            <meta:Height>{h}</meta:Height>
          </meta:MainPictureActiveArea>
          <meta:ExtensionMetadataList>
            <meta:ExtensionMetadata scope="http://isdcf.com/ns/cplmd/app">
              <meta:Name>Application</meta:Name>
              <meta:PropertyList>
                <meta:Property>
                  <meta:Name>DCP Constraints Profile</meta:Name>
                  <meta:Value>SMPTE-RDD-52:2020-Bv2.1</meta:Value>
                </meta:Property>
              </meta:PropertyList>
            </meta:ExtensionMetadata>
          </meta:ExtensionMetadataList>
        </meta:CompositionMetadataAsset>
      </AssetList>
    </Reel>
  </ReelList>
</CompositionPlaylist>
'''
    cpl_name = f'cpl_{cpl_id}.xml'
    with open(os.path.join(a.out, cpl_name), 'w') as f:
        f.write(cpl)

    assets = [(cpl_id, cpl_name, 'text/xml'), (mxf_id, mxf_name, 'application/mxf')]
    pkl_assets = ''
    for aid, name, typ in assets:
        p = os.path.join(a.out, name)
        pkl_assets += f'''    <Asset>
      <Id>urn:uuid:{aid}</Id>
      <AnnotationText>{name}</AnnotationText>
      <Hash>{sha1_b64(p)}</Hash>
      <Size>{os.path.getsize(p)}</Size>
      <Type>{typ}</Type>
      <OriginalFileName>{name}</OriginalFileName>
    </Asset>
'''
    pkl = f'''<?xml version="1.0" encoding="UTF-8"?>
<PackingList xmlns="http://www.smpte-ra.org/schemas/429-8/2007/PKL">
  <Id>urn:uuid:{pkl_id}</Id>
  <AnnotationText>{a.title}</AnnotationText>
  <IssueDate>{now}</IssueDate>
  <Issuer>Gail</Issuer>
  <Creator>Gail GPU JPEG 2000 encoder</Creator>
  <AssetList>
{pkl_assets}  </AssetList>
</PackingList>
'''
    pkl_name = f'pkl_{pkl_id}.xml'
    with open(os.path.join(a.out, pkl_name), 'w') as f:
        f.write(pkl)

    am_assets = ''
    for aid, name, is_pkl in [(pkl_id, pkl_name, True), (cpl_id, cpl_name, False), (mxf_id, mxf_name, False)]:
        am_assets += f'''    <Asset>
      <Id>urn:uuid:{aid}</Id>
{"      <PackingList>true</PackingList>" + chr(10) if is_pkl else ""}      <ChunkList>
        <Chunk>
          <Path>{name}</Path>
          <VolumeIndex>1</VolumeIndex>
          <Offset>0</Offset>
          <Length>{os.path.getsize(os.path.join(a.out, name))}</Length>
        </Chunk>
      </ChunkList>
    </Asset>
'''
    am = f'''<?xml version="1.0" encoding="UTF-8"?>
<AssetMap xmlns="http://www.smpte-ra.org/schemas/429-9/2007/AM">
  <Id>urn:uuid:{am_id}</Id>
  <Creator>Gail GPU JPEG 2000 encoder</Creator>
  <VolumeCount>1</VolumeCount>
  <IssueDate>{now}</IssueDate>
  <Issuer>Gail</Issuer>
  <AssetList>
{am_assets}  </AssetList>
</AssetMap>
'''
    with open(os.path.join(a.out, 'ASSETMAP.xml'), 'w') as f:
        f.write(am)
    with open(os.path.join(a.out, 'VOLINDEX.xml'), 'w') as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n'
                '<VolumeIndex xmlns="http://www.smpte-ra.org/schemas/429-9/2007/AM">\n  <Index>1</Index>\n</VolumeIndex>\n')
    print(f'DCP written to {a.out}: {n} frames {w}x{h} @ {a.fps} fps, picture {mxf_name}')


if __name__ == '__main__':
    main()
