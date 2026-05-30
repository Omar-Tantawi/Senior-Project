<?php

namespace App\Providers;

use App\Services\Transport\DemoLoopProvider;
use App\Services\Transport\PositionProvider;
use App\Services\Transport\RealPingProvider;
use App\Services\Transport\ScheduledSimulationProvider;
use Illuminate\Support\ServiceProvider;

class AppServiceProvider extends ServiceProvider
{
    public function register(): void
    {
        // Bind the live-tracking provider based on TRANSPORT_MODE.
        $this->app->singleton(PositionProvider::class, function ($app) {
            return match (config('transport.mode', 'demo')) {
                'demo'       => $app->make(DemoLoopProvider::class),
                'simulation' => $app->make(ScheduledSimulationProvider::class),
                default      => $app->make(RealPingProvider::class),
            };
        });
    }

    public function boot(): void
    {
        //
    }
}
