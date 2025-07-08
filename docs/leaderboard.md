---
title: Rank-sum Leaderboard
---

# 🏆 Rank-sum Leaderboard

<div class="table-responsive-sm">
  <table id="rankTable" class="table table-hover table-bordered table-sm nowrap w-100">
    <caption>Lower Rank-sum = better overall performance</caption>
    <thead class="align-middle text-center">
      <tr>
        <th>Model</th>
        <th>Domain</th>
        <th>Type</th>
        <th>KNN &uarr;</th>
        <th>Lin. prob. &uarr;</th>
        <th>Few-shot &uarr;</th>
        <th>Calib. &darr;</th>
        <th>Adv. attack &darr;</th>
        <th>Seg.&uarr;</th>
        <th>Rank sum &darr;</th>
      </tr>
    </thead>
    <tbody>
      <tr><td>HIBOU-B</td><td>Histopathology</td><td>VM</td><td>75.8 (10)</td><td>78.0 (14)</td><td>74.2 (6)</td><td>3.7 (2)</td><td>52.8 (14)</td><td>61.5 (10)</td><td>56 (8)</td></tr>
      <tr><td>HIBOU-L</td><td>Histopathology</td><td>VM</td><td>75.2 (12)</td><td>81.2 (7)</td><td>70.4 (12)</td><td>5.5 (21)</td><td>40.0 (5)</td><td>62.2 (7)</td><td>64 (10)</td></tr>
      <tr><td>H-OPTIMUS-0</td><td>Histopathology</td><td>VM</td><td>79.2 (5)</td><td>81.4 (5)</td><td>73.4 (7)</td><td>4.7 (12)</td><td>44.2 (9)</td><td>59.0 (13)</td><td>51 (7)</td></tr>
      <tr><td>H-OPTIMUS-1</td><td>Histopathology</td><td>VM</td><td>80.5 (3)</td><td>83.3 (2)</td><td>74.8 (4)</td><td>4.1 (4)</td><td>58.0 (17)</td><td>58.4 (14)</td><td>44 (5)</td></tr>
      <tr><td>MIDNIGHT</td><td>Histopathology</td><td>VM</td><td>78.2 (8)</td><td>82.9 (3)</td><td>70.6 (11)</td><td>3.2 (1)</td><td>36.3 (4)</td><td>62.9 (2)</td><td>29 (3)</td></tr>
      <tr><td>PHIKON</td><td>Histopathology</td><td>VM</td><td>72.8 (14)</td><td>78.5 (13)</td><td>72.2 (10)</td><td>6.4 (22)</td><td>34.4 (3)</td><td>61.7 (9)</td><td>71 (13)</td></tr>
      <tr><td>PHIKON2</td><td>Histopathology</td><td>VM</td><td>70.1 (15)</td><td>76.5 (15)</td><td>70.1 (13)</td><td>4.6 (10)</td><td>45.6 (11)</td><td>62.0 (8)</td><td>72 (14)</td></tr>
      <tr><td>UNI</td><td>Histopathology</td><td>VM</td><td>78.8 (6)</td><td>81.3 (6)</td><td>76.4 (2)</td><td>4.3 (7)</td><td>42.8 (7)</td><td>62.2 (6)</td><td>34 (4)</td></tr>
      <tr><td>UNI2-H</td><td>Histopathology</td><td>VM</td><td>81.7 (1)</td><td>83.9 (1)</td><td>78.4 (1)</td><td>4.5 (8)</td><td>34.3 (2)</td><td>62.8 (4)</td><td>17 (1)</td></tr>
      <tr><td>VIRCHOW</td><td>Histopathology</td><td>VM</td><td>74.2 (13)</td><td>80.2 (11)</td><td>68.5 (15)</td><td>5.5 (19)</td><td>41.0 (6)</td><td>62.7 (5)</td><td>69 (11)</td></tr>
      <tr><td>VIRCHOW2</td><td>Histopathology</td><td>VM</td><td>81.2 (2)</td><td>82.7 (4)</td><td>72.5 (9)</td><td>4.6 (11)</td><td>33.6 (1)</td><td>63.4 (1)</td><td>28 (2)</td></tr>
      <tr><td>CONCH</td><td>Histopathology</td><td>VLM</td><td>77.3 (9)</td><td>80.2 (10)</td><td>73.1 (8)</td><td>4.3 (6)</td><td>55.0 (15)</td><td>42.0 (23)</td><td>71 (12)</td></tr>
      <tr><td>CONCH&nbsp;1.5</td><td>Histopathology</td><td>VLM</td><td>78.6 (7)</td><td>80.8 (9)</td><td>74.6 (5)</td><td>4.9 (15)</td><td>75.3 (23)</td><td>62.8 (3)</td><td>62 (9)</td></tr>
      <tr><td>KEEP</td><td>Histopathology</td><td>VLM</td><td>79.7 (4)</td><td>81.1 (8)</td><td>75.8 (3)</td><td>4.7 (13)</td><td>44.7 (10)</td><td>61.3 (11)</td><td>49 (6)</td></tr>
      <tr><td>MUSK</td><td>Histopathology</td><td>VLM</td><td>75.6 (11)</td><td>79.0 (12)</td><td>70.0 (14)</td><td>4.5 (9)</td><td>69.3 (22)</td><td>59.7 (12)</td><td>80 (15)</td></tr>
      <tr><td>PLIP</td><td>Histopathology</td><td>VLM</td><td>67.8 (19)</td><td>71.0 (21)</td><td>63.4 (17)</td><td>4.9 (14)</td><td>56.9 (16)</td><td>54.0 (21)</td><td>108 (20)</td></tr>
      <tr><td>QUILTNET</td><td>Histopathology</td><td>VLM</td><td>68.3 (17)</td><td>71.0 (22)</td><td>65.7 (16)</td><td>7.0 (23)</td><td>52.7 (13)</td><td>54.3 (20)</td><td>111 (21)</td></tr>
      <tr><td>DINOv2-B</td><td>Natural-image</td><td>VM</td><td>67.9 (18)</td><td>74.8 (17)</td><td>61.0 (18)</td><td>5.5 (20)</td><td>65.8 (20)</td><td>54.7 (18)</td><td>111 (22)</td></tr>
      <tr><td>DINOv2-L</td><td>Natural-image</td><td>VM</td><td>69.5 (16)</td><td>75.3 (16)</td><td>59.2 (19)</td><td>5.3 (17)</td><td>64.5 (19)</td><td>54.4 (19)</td><td>106 (19)</td></tr>
      <tr><td>ViT-B/16</td><td>Natural-image</td><td>VM</td><td>64.4 (21)</td><td>71.9 (19)</td><td>57.8 (21)</td><td>3.9 (3)</td><td>46.8 (12)</td><td>57.9 (15)</td><td>93 (16)</td></tr>
      <tr><td>ViT-L/16</td><td>Natural-image</td><td>VM</td><td>67.5 (20)</td><td>72.8 (18)</td><td>56.5 (22)</td><td>5.0 (16)</td><td>44.1 (8)</td><td>51.9 (22)</td><td>99 (17)</td></tr>
      <tr><td>CLIP-B/32</td><td>Natural-image</td><td>VLM</td><td>61.9 (23)</td><td>65.8 (23)</td><td>53.3 (23)</td><td>5.5 (18)</td><td>60.4 (18)</td><td>55.9 (16)</td><td>127 (23)</td></tr>
      <tr><td>CLIP-L/14</td><td>Natural-image</td><td>VLM</td><td>64.2 (22)</td><td>71.3 (20)</td><td>58.2 (20)</td><td>4.2 (5)</td><td>67.8 (21)</td><td>55.5 (17)</td><td>104 (18)</td></tr>
    </tbody>
  </table>
</div>
