<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public function up(): void
    {
        Schema::create('conversations', function (Blueprint $table) {
            $table->id();
            $table->foreignId('teacher_user_id')
                  ->constrained('users')
                  ->cascadeOnDelete();
            $table->foreignId('parent_user_id')
                  ->constrained('users')
                  ->cascadeOnDelete();
            $table->timestamp('last_message_at')->nullable();
            $table->timestamps();

            // One conversation per teacher–parent pair.
            $table->unique(['teacher_user_id', 'parent_user_id']);
        });
    }

    public function down(): void
    {
        Schema::dropIfExists('conversations');
    }
};
