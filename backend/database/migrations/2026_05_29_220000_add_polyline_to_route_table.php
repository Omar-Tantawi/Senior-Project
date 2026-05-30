<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

/**
 * Stores a road-snapped polyline per route so the demo bus follows real
 * streets instead of straight lines between stops.
 *
 * Populated by `php artisan transport:fetch-polylines` via OSRM.
 */
return new class extends Migration {
    public function up(): void
    {
        Schema::table('route', function (Blueprint $table) {
            // JSON array of [lat, lng] pairs.
            $table->json('polyline')->nullable()->after('name');
        });
    }

    public function down(): void
    {
        Schema::table('route', function (Blueprint $table) {
            $table->dropColumn('polyline');
        });
    }
};
