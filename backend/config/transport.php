<?php

return [

    /*
    |--------------------------------------------------------------------------
    | Transport Mode
    |--------------------------------------------------------------------------
    |
    |   demo        – Always-moving virtual loop. Ignores wall clock and
    |                 calendar. Best for university demos / vivas.
    |   simulation  – Wall-clock driven school-day simulation
    |                 (morning + afternoon windows). Reserved for Stage 4.
    |   real        – Only real GPS pings from the driver app.
    |
    */
    'mode' => env('TRANSPORT_MODE', 'demo'),

    'demo' => [
        // Full Stop 1 → School → Stop 1 round trip duration, in seconds.
        'loop_seconds'      => (int) env('TRANSPORT_DEMO_LOOP_SECONDS', 120),
        // Per-bus phase offset so multiple buses don't sit on top of each other.
        'bus_phase_offset'  => (int) env('TRANSPORT_DEMO_BUS_PHASE_OFFSET', 30),
    ],

    'simulation' => [
        'morning_window'    => env('TRANSPORT_MORNING_WINDOW',   '07:00-07:50'),
        'afternoon_window'  => env('TRANSPORT_AFTERNOON_WINDOW', '13:30-14:20'),
    ],

    'real' => [
        // If the latest ping is older than this, surface the bus as stale.
        'ping_stale_seconds' => (int) env('TRANSPORT_REAL_PING_STALE_SECONDS', 120),
    ],

];
