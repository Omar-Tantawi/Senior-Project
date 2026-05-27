<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public function up(): void
    {
        Schema::create('classdistribution', function (Blueprint $table) {
            $table->string('distribution_id', 36)->primary();
            $table->unsignedInteger('class_id')->nullable();
            $table->integer('num_classes');
            $table->json('criteria_weights');
            $table->double('balance_score')->nullable();
            $table->boolean('is_balanced')->nullable();
            $table->text('explanation')->nullable();
            $table->string('status', 20)->default('draft');
            $table->timestamps();
        });

        Schema::create('classassignment', function (Blueprint $table) {
            $table->increments('assignment_id');
            $table->string('distribution_id', 36);
            $table->unsignedBigInteger('student_id');
            $table->string('class_name', 50);
            $table->timestamps();
        });
    }

    public function down(): void
    {
        Schema::dropIfExists('classassignment');
        Schema::dropIfExists('classdistribution');
    }
};
